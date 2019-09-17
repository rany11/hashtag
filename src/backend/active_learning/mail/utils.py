import base64
from email.mime.text import MIMEText

from mail.exceptions import HeaderNotFoundError


class Thread:
    @staticmethod
    def get_id(thread):
        return thread['id']

    @staticmethod
    def get_messages(thread):
        return thread['messages']


class Message:
    TO = 'To'
    FROM = 'From'
    SUBJECT = 'Subject'
    MSG_ID = 'Message-ID'
    REFERENCES = 'References'
    IN_REPLY_TO = 'In-Reply-To'
    THREAD_ID = 'threadId'

    @staticmethod
    def __get_payload(msg):
        return msg['payload']

    @staticmethod
    def __get_headers(msg):
        return Message.__get_payload(msg)['headers']

    @staticmethod
    def __get_header(msg, header_name):
        for header in Message.__get_headers(msg):
            if header['name'] == header_name:
                return header['value']
        raise HeaderNotFoundError('msg has no header {}'.format(header_name))

    @staticmethod
    def get_id(msg):
        return msg['id']

    @staticmethod
    def get_thread_id(msg):
        return msg['threadId']

    @staticmethod
    def get_subject(msg):
        return Message.__get_header(msg, 'Subject')

    @staticmethod
    def get_body(msg):
        return Message.__get_payload(msg)['body']  # TODO: decode from base64

    @staticmethod
    def get_snippet(msg):
        return msg['snippet']

    @staticmethod
    def get_labels(msg):
        return msg['labelIds']

    @staticmethod
    def create_message(to, subject, message_text, thread_id='', sender='', references='', in_reply_to=''):
        """Create a message for an email.
        :param thread_id: the id of the thread of the mail
        :param sender: email address of the sender
        :param to: email address of the receiver
        :param subject: the subject of the email
        :param message_text: the body of the email
        :param references: the id of the thread the mail references
        :param in_reply_to: the id of the message the mail replies to
        :return: an object containing a base64url encoded email object.
        """
        message = MIMEText(message_text)
        message[Message.THREAD_ID] = thread_id
        message[Message.TO] = to
        message[Message.FROM] = sender
        message[Message.SUBJECT] = subject
        message[Message.REFERENCES] = references
        message[Message.IN_REPLY_TO] = in_reply_to
        return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

    @staticmethod
    def to_string(msg):
        payload = Message.__get_payload(msg)
        if payload['mimeType'].startswith('multipart'):
            return base64.b64decode(payload['parts'][0]['body']['data'])

        else:
            return Message.get_snippet(msg)

    @staticmethod
    def is_outgoing(msg):
        return "SENT" in Message.get_labels(msg)
