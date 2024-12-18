import streamlit as st
import imaplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification,AutoModelForSequenceClassification,AutoTokenizer
from datetime import datetime,timedelta,timezone
import pandas as pd
import json
import boto3
import smtplib
from dotenv import load_dotenv
import os
import requests
from groq import Groq
from supabase import Client,create_client

#initialize the bedrock client for claude
bedrock_runtime = boto3.client('bedrock-runtime', region_name='ap-south-1')
load_dotenv()

#creating supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_API_KEY = os.getenv('SUPABASE_API_KEY')
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)


# Load the pre-trained DistilBERT classification model
model_path = "Michael444/DistilbertModel"  # Path to the saved model
loaded_model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
##############

# Streamlit session state initialization
if "imap_session" not in st.session_state:
    st.session_state.imap_session = None

if "cached_emails" not in st.session_state:
    st.session_state.cached_emails = []

if "replied_emails" not in st.session_state:
    st.session_state.replied_emails = set()

# App title
st.title("Email Client with AI Classification & Storage")

# Sidebar for user login
st.sidebar.header("Login")
email_user = st.sidebar.text_input("Email Address", placeholder="example@gmail.com")
email_pass = st.sidebar.text_input("App Password", type="password", placeholder="Your App Password")

# Function to insert email data into the database
def save_email_to_db(user_email, email_data):
    # Prepare the data to be inserted into the Supabase table
    email = {
        'email_id': email_data['email_id'],
        'user_email': user_email,
        'sender': email_data['from'],
        'subject': email_data['subject'],
        'content': email_data['content'],
        'arrivedat': email_data['arrivedat'],
        'date': email_data['date'],
        'is_important': int(email_data['is_important']),  # Classify importance
        'status':0,
    }

    # Insert data into the Supabase table
    response = supabase.table('emails3').insert(email).execute()

# Function to fetch emails from the database for today only

def fetch_emails_from_db_for_today(user_email):
    # Get today's date in the required format
    today = datetime.today().strftime("%d-%b-%Y")
    formatted_date1 = today  # Example: "17-Dec-2024"

    # Debugging output
    print(f"Formatted Date 1: {formatted_date1}")

    # Query the Supabase table for today's emails
    response = supabase.table("emails3") \
                       .select("*") \
                       .filter("user_email", "eq", user_email) \
                       .filter("date", "eq", formatted_date1) \
                       .order("email_id", desc=True) \
                       .execute()

    # Access the data and error attribute

    rows = response.data  # Emails data

    # Debugging: Print the results of the query
    if not rows:
        print(f"No emails found for {formatted_date1} in the Supabase database.")
    else:
        print(f"Found {len(rows)} emails for {formatted_date1}.")

    # Convert rows into a list of dictionaries (optional)
    emails = []
    for row in rows:
        emails.append({
            "email_id": row["email_id"],
            "from": row["sender"],
            "subject": row["subject"],
            "content": row["content"],
            "ArrivedAt": row["arrivedat"],
            "Date": row["date"],
            "is_important": row["is_important"]
        })

    return emails

# Function to classify importance of emails
def is_important(message_text):
    inputs = tokenizer(message_text, return_tensors="tf", truncation=True, padding=True)
    outputs = loaded_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    prediction = tf.argmax(outputs.logits, axis=-1).numpy().item()
    return prediction == 1

# Connect to the email server (IMAP for fetching emails)
def connect_to_email(user, password):
    try:
        if st.session_state.imap_session is None:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(user, password)
            st.session_state.imap_session = mail
            st.success("Logged in successfully!")
        else:
            st.success("Reusing existing email session.")
        return st.session_state.imap_session
    except Exception as e:
        st.error(f"Failed to connect: {str(e)}")
        return None

# Fetch new emails for today
def fetch_emails(mail, folder="INBOX", num_emails=10):
    try:
        mail.select(folder)
        
        # Fetch only emails from today
        today = datetime.today().strftime("%d-%b-%Y")
        _, data = mail.search(None, f'SINCE {today}')
        
        email_ids = data[0].split()[-num_emails:]  # Get the last num_emails emails
        emails = []

        for email_id in reversed(email_ids):
            email_id_decoded = email_id.decode()
            response = supabase.table("emails3").select("email_id").eq("email_id", email_id_decoded).execute()
            if response.data:
                #print(f"Email with ID {email_id_decoded} is already present in the database. Skipping...")
                continue
            #If not found, process this email
            _, msg_data = mail.fetch(email_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])

                    # Decode subject
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or "utf-8", errors="ignore")

                    # Extract email details
                    from_ = msg.get("From")
                    email_date = msg.get("Date")
                    email_body = display_email_content(msg)

                    # Debugging - Print parsed details
                    st.write(f"Subject: {subject}, From: {from_}, Date: {email_date}")

                    # Classify email and save to DB
                    email_data = {
                        "email_id": email_id.decode(),
                        "from": from_,
                        "subject": subject,
                        "content": email_body,
                        "arrivedat": email_date,
                        "date":today,
                        "is_important": is_important(email_body),
                        'status':0
                    }
                    save_email_to_db(email_user, email_data)
                    emails.append(email_data)

        return emails

    except Exception as e:
        st.error(f"Error fetching emails: {str(e)}")
        return []

# Function to display email content
def display_email_content(msg):
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                return part.get_payload(decode=True).decode()
    else:
        return msg.get_payload(decode=True).decode()

# Display emails
def display_emails(emails):
    if not emails:
        st.write("No emails to display.")
        return

    for idx, email_data in enumerate(emails):
        email_id = email_data.get('email_id')
        if email_id in st.session_state.replied_emails:
            continue

        st.write(f"Email #{idx + 1} from: {email_data.get('from', 'Unknown')}")
        st.write(f"Subject: {email_data.get('subject', 'No Subject')}")
        st.write(f"Content: {email_data.get('content', 'No Content')}")
        st.write(f"Date: {email_data.get('ArrivedAt', 'Unknown Date')}")

        if email_data.get("is_important", 0):
            st.write("The email is classified as IMPORTANT.")
        else:
            st.write("The email is classified as NOT important.")
        st.write(f"###############################################################################")
        st.write(f"-------------------------------------------------------------------------------")
        st.write(f"###############################################################################")
# App initialization
#setup_database()


def fetch_important_emails(user_email):
    # Get today's date in the required format
    today = datetime.today().strftime("%d-%b-%Y")
    formatted_date1 = today  # Example: "17-Dec-2024"
    one =1
    # Debugging output
    print(f"Formatted Date 1: {formatted_date1}")

    # Query the Supabase table for today's emails
    response = supabase.table("emails3") \
                        .select("*") \
                        .filter("user_email", "eq", user_email) \
                        .filter("date", "eq", today) \
                        .filter("is_important", "eq", one) \
                        .order("email_id", desc=True) \
                        .execute()

    # Access the data and error attribute

    rows = response.data  # Emails data

    # Debugging: Print the results of the query
    if not rows:
        print(f"No emails found for {formatted_date1} in the Supabase database.")
    else:
        print(f"Found {len(rows)} emails for {formatted_date1}.")

    # Convert rows into a list of dictionaries (optional)
    emails = []
    for row in rows:
        emails.append({
            "email_id": row["email_id"],
            "from": row["sender"],
            "subject": row["subject"],
            "content": row["content"],
            "ArrivedAt": row["arrivedat"],
            "Date": row["date"],
            "is_important": row["is_important"]
        })

    return emails

def display_emails_final(emails):
    if not emails:
        st.write("No emails to display.")
        return

    # Create a list of dictionaries for tabular representation
    email_data_list = []
    for email_data in emails:
        email_id = email_data.get('email_id')
        if email_id in st.session_state.replied_emails:
            continue

        email_data_list.append({
            "Sender": email_data.get('from', 'Unknown'),
            "Subject": email_data.get('subject', 'No Subject'),
            "Content": email_data.get('content', 'No Content'),
            "ArrivedAt": email_data.get('ArrivedAt', 'Unknown Date'),
            "Date": email_data.get('Date', 'Unknown Date'),
            "Importance": "Important" if email_data.get("is_important", 1) else "Not Important",
        })

    # Convert the list of dictionaries to a Pandas DataFrame
    email_df = pd.DataFrame(email_data_list)

    # Display the DataFrame using Streamlit's table or dataframe methods
    if not email_df.empty:
        st.write("### Today's Emails")
        st.dataframe(email_df)  # For a scrollable, interactive table
    else:
        st.write("No emails to display in the table.")


def get_claude_response(email_body):
    """Fetch a human-like automated response from Claude model based on the email body."""
    prompt = f"""You are a human replying to an email, and your response should feel warm, genuine, and tailored to the specific message received. Below is the content of the email you’re replying to:

    {email_body}

    Please compose a thoughtful, human-like response based on the following guidelines:

    1. **Empathy and Friendliness**: Your tone should be friendly, approachable, and empathetic. Avoid overly formal language; instead, aim for a tone that feels natural and conversational.
    2. **Context Acknowledgment**: Address specific details and respond meaningfully.
    3. **Clarity and Brevity**: Make the response concise and clear.

    Now, please write a short, thoughtful reply."""
    # Bedrock payload
    kwargs = {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }),
    }
    # Send request to Bedrock
    response = bedrock_runtime.invoke_model(**kwargs)
    body = json.loads(response['body'].read())
    return body.get("content", [{}])[0].get("text", "Error generating response")




def get_llama_response(email_body):
    """Fetch a human-like automated response from the llama3-8b-8192 model using the Groq API."""
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("Groq API key is missing from environment variables.")

    prompt = f"""You are a human replying to an email, and your response should feel warm, genuine, and tailored to the specific message received. Below is the content of the email you’re replying to:

    {email_body}

    Please compose a thoughtful, human-like response based on the following guidelines:

    1. **Empathy and Friendliness**: Your tone should be friendly, approachable, and empathetic. Avoid overly formal language; instead, aim for a tone that feels natural and conversational.
    2. **Context Acknowledgment**: Address specific details and respond meaningfully.
    3. **Clarity and Brevity**: Make the response concise and clear.

    Now, please write a short, thoughtful reply."""

    # Initialize Groq client
    client = Groq(api_key=groq_api_key)

    try:
        # Groq API request to generate response
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
            top_p=1.0,
            stream=False  # Not streaming in this case
        )

        # Debugging: Print the structure of the completion object to inspect
        print("Full response from Groq API:", completion)

        # Assuming `choices` is a list of ChatCompletionMessage objects
        # Accessing the message content properly:
        # If 'message' is an object, use its attribute, such as message.content
        message_content = completion.choices[0].message.content
        return message_content  # Return the response content

    except Exception as e:
        print(f"Exception: {str(e)}")
        return f"Error: {str(e)}"


def send_email(from_email, app_password, to_email, subject, body):
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(from_email, app_password)

        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False



def send_automated_replies1():
    # Database connection
    conn = sqlite3.connect('emails3.db')
    cursor = conn.cursor()

    # Fetch only today's important emails where status is 0
    today = datetime.today().strftime("%d-%b-%Y")
    query = """
        SELECT email_id, sender, subject, content
        FROM emails3
        WHERE is_important = 1 AND status = 0 AND Date = ?
    """
    cursor.execute(query, (today,))
    emails = cursor.fetchall()

    if not emails:
        st.info("No important emails found for today that need replies.")
        return

    # Process each email
    st.info(f"Found {len(emails)} important emails to reply to.")
    for email_id, sender, subject, content in emails:
        st.write(f"Generating reply for email from {sender}...")
        
        # Generate response using Claude
        #response = get_claude_response(content)
        response = get_llama_response(content)
        if response.startswith("Error"):
            st.error(f"Failed to generate response for email {email_id}.")
            continue
        
        # Send the email
        sent = send_email(
            from_email=email_user,  # Update your email
            app_password=email_pass,  # Update your app password
            to_email=sender,
            subject=f"Re: {subject}",
            body=response
        )

        if sent:
            # Update the status to 1 (reply sent)
            update_query = "UPDATE emails3 SET status = 1 WHERE email_id = ?"
            cursor.execute(update_query, (email_id,))
            conn.commit()
            st.success(f"Reply sent to {sender} and status updated.")
        else:
            st.error(f"Failed to send reply to {sender}.")

    conn.close()


def send_automated_replies():
    # Fetch only today's important emails where status is 0
    today = datetime.today().strftime("%d-%b-%Y")
    
    # Query Supabase to get today's important emails with status = 0
    response = supabase.table("emails3") \
        .select("email_id, sender, subject, content") \
        .filter("is_important", "eq", 1) \
        .filter("status", "eq", 0) \
        .filter("date", "eq", today) \
        .execute()



    emails = response.data

    if not emails:
        st.info("No important emails found for today that need replies.")
        return

    # Process each email
    st.info(f"Found {len(emails)} important emails to reply to.")
    for email in emails:
        email_id = email["email_id"]
        sender = email["sender"]
        subject = email["subject"]
        content = email["content"]

        st.write(f"Generating reply for email from {sender}...")

        # Generate response using Claude or LLaMA
        response = get_llama_response(content)
        if response.startswith("Error"):
            st.info(f"Failed to generate response for email {email_id}.")
            continue

        # Send the email
        sent = send_email(
            from_email=email_user,  # Replace with your email
            app_password=email_pass,  # Replace with your app password
            to_email=sender,
            subject=f"Re: {subject}",
            body=response
        )

        if sent:
            # Update the status of the email in Supabase to 1 (reply sent)
            update_response = supabase.table("emails3") \
                .update({"status": 1}) \
                .eq("email_id", email_id) \
                .execute()

            st.info(f"Reply sent to {sender} and status updated.")
        else:
            st.info(f"Failed to send reply to {sender}.")


def send_manual_replies():
    # Database connection
    conn = sqlite3.connect('emails3.db')
    cursor = conn.cursor()
    # Fetch only today's important emails where status is 0
    today = datetime.today().strftime("%d-%b-%Y")
    query = """
        SELECT email_id, sender, subject, content
        FROM emails3
        WHERE is_important = 1 AND status = 0 AND Date = ? 
        ORDER BY email_id DESC
    """
    cursor.execute(query, (today,))
    emails = cursor.fetchall()

    if not emails:
        st.info("No important emails found for today that need replies.")
        return

    st.info(f"Found {len(emails)} important emails to review and reply to.")

    # Process each email
    for email_id, sender, subject, content in emails:
        st.write(f"### Email from {sender} - Subject: {subject}")
        st.write(f"**Content:** {content}")

        # Generate response dynamically
        response = get_llama_response(content)
        if response.startswith("Error"):
            st.error(f"Failed to generate response for email {email_id}.")
            continue

        # Allow user to edit the generated response
        edited_response = st.text_area(
            "Edit the Reply:", 
            value=response,  # Set initial value
            height=150, 
            key=f"response_{email_id}_editable"  # Unique key for each email
        )

        # Manual send button
        if st.button(f"Send Reply to {sender}", key=f"send_{email_id}"):
            # Send the email with the edited response
            sent = send_email(
                from_email=email_user,  # Replace with your email
                app_password=email_pass,    # Replace with your app password
                to_email=sender,
                subject=f"Re: {subject}",
                body=edited_response
            )
            if sent:
                # Update the database status to 1 (reply sent)
                update_query = "UPDATE emails3 SET status = 1 WHERE email_id = ?"
                cursor.execute(update_query, (email_id,))
                conn.commit()
                st.success(f"Reply sent to {sender} and status updated.")
            else:
                st.error(f"Failed to send reply to {sender}.")

        # Add a visual divider
        st.markdown("---")

    # Close the database connection
    conn.close()

# Main app logic
if st.sidebar.button("Login"):
    if email_user and email_pass:
        mail = connect_to_email(email_user, email_pass)
        if mail:
            if not st.session_state.cached_emails:
                emails = fetch_emails(mail)
                st.session_state.cached_emails.extend(emails)
                st.info(f"Fetched {len(emails)} new emails.")
            else:
                st.success("Using cached emails.")

            # Load today's emails from the database
            today_emails = fetch_emails_from_db_for_today(email_user)
            st.session_state.cached_emails.extend(today_emails)
    else:
        st.error("Please enter both email and password.")

# Button to show emails already stored in the database
if st.sidebar.button("Show Today's All Emails"):
    today_emails = fetch_emails_from_db_for_today(email_user)
    if today_emails:
        display_emails_final(today_emails)
    else:
        st.info("No emails found for today in the database.")
if st.sidebar.button("Show Today's important Emails Only "):
    important_emails = fetch_important_emails(email_user)
    if important_emails:
        display_emails_final(important_emails)
    else:
        st.info("No emails found for today in the database.")
        
if st.sidebar.button("Send Automated Reply To Today's Important Emails"):
    send_automated_replies()


if st.sidebar.button("Send Manual Reply"):
    send_manual_replies()
    print("testing 2")

# Logout button
if st.sidebar.button("Logout"):
    st.session_state.imap_session = None
    st.session_state.cached_emails = []
    st.session_state.replied_emails = set()
    st.experimental_rerun()
