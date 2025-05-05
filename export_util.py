import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mplcyberpunk
import fitz
import numpy as np
#from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import qrcode
import httplib2
import os
import oauth2client
from oauth2client import client, tools, file
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from apiclient import errors, discovery
import mimetypes
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from datetime import datetime

SCOPES = 'https://www.googleapis.com/auth/gmail.send'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Gmail API Python Send Email'

def generate_graph(filename):
    # data1 = pd.read_csv('CSV_data/'+filename, dtype={'Time': str, 'FER': str, 'POSE': str, 'SER': str, 'BPM' : str, 'Systolic' : str, 'Diastolic' : str})
    data1 = pd.read_csv('CSV_data/'+filename, dtype={'Time': str, 'FER': str, 'POSE': str, 'SER': str})
    emotions = ['Neutral', 'Angry', 'Happy', 'Sad', 'Surprise']
    emotions_labels = ['Neutral', 'Angry', 'Happy', 'Sad', 'Surprise']
    plt.style.use("cyberpunk")
    # plt.rcParams['figure.facecolor'] = '#1f1f1f'
    # plt.rcParams['axes.facecolor'] = '#1f1f1f'
    
    # plt.gca().autoscale(enable=True, axis='x', tight=True)
    # mplcyberpunk.make_lines_glow(plt.gca())

    plt.figure(figsize=(16/2.54, 10/2.54))
    plt.autoscale(enable=True, axis='x')
    plt.grid(True)
    plt.title('Facial Emotion Recognition')
    plt.xlabel('Time')
    plt.ylabel('Emotion')
    plt.plot(list(data1['Time']), list(data1['FER']))
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.xticks(rotation=45)
    plt.gca().set_facecolor('none')
    mplcyberpunk.add_glow_effects()
    plt.yticks(emotions, emotions_labels)
    plt.savefig('Graphs/FER_plot.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.clf()

    plt.figure(figsize=(16/2.54, 10/2.54))
    plt.autoscale(enable=True, axis='x')
    plt.grid(True)
    plt.title('Posture Estimation and Emotion Recognition')
    plt.xlabel('Time')
    plt.ylabel('Emotion')
    plt.plot(list(data1['Time']), list(data1['POSE']))
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.xticks(rotation=45)
    plt.gca().set_facecolor('none')
    mplcyberpunk.add_glow_effects()
    plt.yticks(emotions, emotions_labels)
    plt.savefig('Graphs/POSE_plot.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.clf()

    plt.figure(figsize=(16/2.54, 10/2.54))
    plt.autoscale(enable=True, axis='x')
    plt.grid(True)
    plt.title('Voice Emotion Recognition')
    plt.xlabel('Time')
    plt.ylabel('Emotion')
    plt.plot(list(data1['Time']), list(data1['SER']))
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.xticks(rotation=45)
    plt.gca().set_facecolor('none')
    mplcyberpunk.add_glow_effects()
    plt.yticks(emotions, emotions_labels)
    plt.savefig('Graphs/SER_plot.png', transparent=True, bbox_inches='tight', pad_inches=0)
    plt.clf()

def generate_report(name, fer_text, pose_text, ser_text, phy_text, summary_text):
    pdf_document = fitz.open('Report/template/test_report.pdf')

    x_position_img = 65  
    y_position_img = 120 
    width = 480  
    height = 400
    x_position_text = 45
    y_position_text = 530
    image_rect = fitz.Rect(x_position_img, y_position_img, x_position_img + width, y_position_img + height)
    text_rect = fitz.Rect(x_position_text, y_position_text, x_position_text + 450, y_position_text + 300)

    custom_font_path = "TT-Interphases-R.ttf"

    cover_page = pdf_document[0]
    cover_page.wrap_contents()
    image_path='logo.png'
    img = open(image_path, 'rb').read()
    cover_page.insert_image(fitz.Rect(380, 0, 580, 200), stream=img, xref=0 )  
    cover_page.insert_font(fontname="TT", fontfile=custom_font_path)
    cover_page.insert_textbox(fitz.Rect(100, 650, 500, 800), name, fontsize=18, fontname="TT", color=(1,1,1), align=fitz.TEXT_ALIGN_LEFT)
    
    FER_page = pdf_document[2]
    FER_page.wrap_contents()
    image_path='Graphs/FER_plot.png'
    img = open(image_path, "rb").read()
    FER_page.insert_image(image_rect, stream=img, xref=0 )
    FER_page.insert_font(fontname="TT", fontfile=custom_font_path)
    FER_page.insert_textbox(text_rect, fer_text, fontsize=14, fontname="TT", color=(1,1,1), align=fitz.TEXT_ALIGN_JUSTIFY)

    POSE_page = pdf_document[3]
    POSE_page.wrap_contents()
    image_path='Graphs/POSE_plot.png'
    img = open(image_path, "rb").read()
    POSE_page.insert_image(image_rect, stream=img, xref=0 )
    POSE_page.insert_font(fontname="TT", fontfile=custom_font_path)
    POSE_page.insert_textbox(text_rect, pose_text, fontsize=14, fontname="TT", color=(1,1,1), align=fitz.TEXT_ALIGN_JUSTIFY)
    
    SER_page = pdf_document[4]
    SER_page.wrap_contents()
    image_path='Graphs/SER_plot.png'
    img = open(image_path, "rb").read()
    SER_page.insert_image(image_rect, stream=img, xref=0 )
    SER_page.insert_font(fontname="TT", fontfile=custom_font_path)
    SER_page.insert_textbox(text_rect, ser_text, fontsize=14, fontname="TT", color=(1,1,1), align=fitz.TEXT_ALIGN_JUSTIFY)

    # PHY_page = pdf_document[5]
    # PHY_page.wrap_contents()
    # image_path='Graphs/BP_Heart_Rate_plot.png'
    # img = open(image_path, "rb").read()
    # PHY_page.insert_image(image_rect, stream=img, xref=0 )
    # PHY_page.insert_font(fontname="TT", fontfile=custom_font_path)
    # PHY_page.insert_textbox(text_rect, phy_text, fontsize=14, fontname="TT", color=(1,1,1), align=fitz.TEXT_ALIGN_JUSTIFY)

    Summary_page = pdf_document[5]
    Summary_page.wrap_contents()
    Summary_page.insert_font(fontname="TT", fontfile=custom_font_path)
    Summary_page.insert_textbox(fitz.Rect(45, 180, 500, 800), summary_text, fontsize=18, fontname="TT", color=(1,1,1), align=fitz.TEXT_ALIGN_LEFT)
    
    date = datetime.now().strftime("%d-%m-%Y_%H-%M")
    pdf_document.save('Report/'+name+"_"+date+'.pdf')
    pdf_document.close()
    return name+"_"+date+'.pdf'

def Azure_upload(filename):
    pass
    '''storage_account_name = "cosmocareai"
    storage_account_key = "fct2/7HuHEz4Ij6FBgMy2plHlw7yVTSovDR/3z1nVsRW9VBKD1X0XWlTlOTpn+eo31MFCcW7hZFd+ASt2ChLUw=="
    container_name = "report"
    local_file_path = f"Report/{filename}"
    blob_name = f"{filename}.pdf"

    blob_service_client = BlobServiceClient(account_url=f"https://{storage_account_name}.blob.core.windows.net", credential=storage_account_key)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    print('Uploading Report please wait.')
    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data, connection_timeout=1000)
    print(f"File '{local_file_path}' uploaded to '{blob_name}' in '{container_name}'.")
    url =  f"https://{storage_account_name}.blob.core.windows.net/{container_name}/{blob_name}"
    return url'''
    
def QR_gen(url):
    pass
    '''storage_account_name = "cosmocareai"
    storage_account_key = "fct2/7HuHEz4Ij6FBgMy2plHlw7yVTSovDR/3z1nVsRW9VBKD1X0XWlTlOTpn+eo31MFCcW7hZFd+ASt2ChLUw=="
    container_name = "email-assets"
    local_file_path = f"qr.png"
    blob_name = f"qrcode.png"

    blob_service_client = BlobServiceClient(account_url=f"https://{storage_account_name}.blob.core.windows.net", credential=storage_account_key)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.delete_blob()

    qr = qrcode.QRCode(
        version=1,  
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,  
        border=4, 
    )

    qr.add_data(url)
    qr.make(fit=True)

    qr_image = qr.make_image(fill_color="black", back_color="white")

    qr_image.save("qr.png")
    print('Uploading QR please wait.')
    with open(local_file_path, "rb") as data:
        blob_client.upload_blob(data)

    print(f"QR code generated")'''

def get_credentials():
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'gmail-python-email-send.json')
    store = oauth2client.file.Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        credentials = tools.run_flow(flow, store)
        print('Storing credentials to ' + credential_path)
    return credentials

def SendMessage(to, name, url):
    subject = "CosmoCare AI - Report"
    sender = "cosmocareai@gmail.com"
    email_html = open("email.html")
    msgHtml = email_html.read().format_map({'name' : name, 'url' : url})
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)
    message1 = CreateMessageHtml(sender, to, subject, msgHtml, "")
    result = SendMessageInternal(service, "me", message1)
    return result

def SendMessageInternal(service, user_id, message):
    try:
        message = (service.users().messages().send(userId=user_id, body=message).execute())
        print('Message Id: %s' % message['id'])
        return message
    except errors.HttpError as error:
        print('An error occurred: %s' % error)
        return "Error"
    return "OK"

def CreateMessageHtml(sender, to, subject, msgHtml, msgPlain):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to
    msg.attach(MIMEText(msgPlain, 'plain'))
    msg.attach(MIMEText(msgHtml, 'html'))
    return {'raw': base64.urlsafe_b64encode(msg.as_string().encode('utf-8')).decode('utf-8')}

def content(analysis, state):
    if state == "Normal":
        if analysis == 'FER':
            return "CosmoCareAI has consistently recognized positive emotions in your facial expressions, reflecting your mental state as generally optimistic and content. This aligns with the observation of emotions associated with good mental health."
        elif analysis == 'POSE':
            return "The AI analysis of your body posture does not indicate any anomalies or stress-related postural changes. Your posture remains consistent with that of an individual in good mental and physical health."
        elif analysis == 'SER':
            return "Your speech patterns indicate a stable and positive emotional state. The absence of signs of anxiety or depression in your voice signals a robust mental well-being."
        elif analysis == 'PHY':
            return "Heart Rate: Your resting heart rate remains within the healthy range of 60 to 80 BPM, indicating a calm and composed state of mind.\n\nHeart Rate Variability: The variability in your heart rate signifies emotional resilience and adaptability, further affirming your positive mental state.\n\nBlood Pressure: Your blood pressure consistently remains within the normal range of 120/80 mmHg, demonstrating a balanced emotional state and overall health."
        elif analysis == 'SUMMARY':
            return "In light of the data collected and the comprehensive analysis conducted by CosmoCareAI, we are pleased to certify that your mental fitness is within the normal range for an astronaut preparing for future space missions. The synchronization between your emotional expressions, voice patterns, body posture, and physiological parameters paints a consistent picture of positive mental well-being. You exhibit emotional resilience, adaptability, and excellent psychological balance, essential qualities for thriving in the demanding environment of space exploration.\n\n\nWe trust that this comprehensive report serves as a testament to your dedication and commitment to maintaining exceptional mental health. Should you require any further support or counseling, our team of experts is readily available to assist you on your journey. Wishing you continued success and a bright future in your endeavors."
        
    elif state == "Depression":
        if analysis == 'FER':
            return "CosmoCareAI has detected a consistent prevalence of sad emotions in your facial expressions, suggesting a shift from the anticipated emotional equilibrium."
        elif analysis == 'POSE':
            return "While the AI analysis of your body posture does not reveal significant abnormalities, it complements the aforementioned emotional indicators, contributing to the overall assessment of your mental well-being."
        elif analysis == 'SER':
            return "In your speech patterns, there is a discernible presence of sadness, as indicated by AI sentiment analysis. This discordance from the expected emotional state is noteworthy."
        elif analysis == 'PHY':
            return "Heart Rate: Your resting heart rate falls below the typical range, exhibiting bradycardia with frequencies below 40 BPM, a hallmark of depressive states. This is further substantiated by the recurrent recognition of sad emotions by the AI.\n\nHeart Rate Variability: The conspicuous absence of heart rate variability, alongside your bradycardic state, collectively underscores the signs of depression.\n\nTachycardia: While your heart rate remains within the normal range during physical activity, sudden spikes in heart rate observed in the absence of physical exertion suggest heightened anxiety levels. This aligns with frequent angry emotion detection by the AI.\n\nBlood Pressure: Your consistently low blood pressure, measuring below 120 mmHg, concurs with the indications of depression, especially when paired with frequent recognition of sad emotions by the AI."
        elif analysis == 'SUMMARY':
            return "In light of the data and analysis provided by CosmoCareAI, we must regrettably report that your mental fitness is displaying deviations from the anticipated norm, signaling potential indicators of depression and anxiety. The convergence of sadness detected in facial expressions, voice analysis, and physiological metrics, including bradycardia and low blood pressure, underscores this concern.\n\nWe highly recommend that you seek immediate support and counseling from our expert mental health professionals. Depression and anxiety are serious conditions, and early intervention is crucial for your well-being and the success of future missions. Our team is committed to providing you with the necessary resources and assistance to address these concerns.\n\nWe understand that the journey to space is demanding, both physically and mentally, and your health is of paramount importance. Please do not hesitate to reach out to our support team for guidance and support as we work together to address these challenges."

    elif state == "Anxiety":
        if analysis == 'FER':
            return "CosmoCareAI has detected a subtle yet consistent prevalence of sad emotions in your facial expressions. While not indicative of depression, it does suggest a departure from the expected emotional equilibrium toward a slightly anxious state."
        elif analysis == 'POSE':
            return "While the AI analysis of your body posture does not reveal significant abnormalities, it complements the aforementioned emotional indicators, contributing to the overall assessment of your mental well-being."
        elif analysis == 'SER':
            return "In your speech patterns, there is a discernible presence of sadness, as indicated by AI sentiment analysis. This departure from the expected emotional state, though subtle, aligns with a slightly anxious disposition."
        elif analysis == 'PHY':
            return "Heart Rate: Your resting heart rate, ranging between 40 to 60 BPM, marginally exceeds the lower threshold. While not indicative of a severe anxiety state, it suggests a slight deviation from the anticipated norm.\n\nHeart Rate Variability: The presence of low heart rate variability, coupled with the elevated resting heart rate, hints at an emerging anxiety pattern, although not yet definitive.\n\nTachycardia: Sudden spikes in heart rate observed in the absence of physical exertion suggest heightened anxiety levels. This aligns with frequent angry emotion detection by the AI.\n\nYour consistently normal blood pressure, measuring below 120 mmHg, does not substantiate depression indications. However, the presence of sad emotions recognized by the AI may signify an emerging anxious disposition."
        elif analysis == 'SUMMARY':
            return "In light of the data and analysis provided by CosmoCareAI, it is noteworthy that your mental fitness is subtly deviating from the expected norm, suggesting the presence of nascent anxiety. While not yet indicative of a severe anxiety state, the convergence of subtle emotional indicators with physiological deviations merits your attention.\n\nWe recommend vigilance and continued monitoring of your mental well-being. Space missions are demanding endeavors, and early recognition of emotional patterns is crucial for your well-being and the success of future missions. Please feel free to reach out to our support team for guidance and assistance as we continue to refine our understanding of these emerging patterns. Your commitment to space exploration is commendable, and we are here to support you in every aspect of your journey."
    else:
        return ""
    
'''if __name__ == '__main__':

    Azure_upload("Sandeep Adithya_24-09-2023_22-45 copy.pdf")'''                                                                                                                                                                              