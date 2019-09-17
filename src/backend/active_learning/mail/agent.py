from __future__ import print_function

from sys import stderr

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from httplib2 import Http
from oauth2client import file, client, tools

# If modifying these scopes, delete the file token.json.
from experimentation.configuration.configuration import Configuration, ConfigurationParsingError
from mail.utils import Message, Thread

SCOPES = 'https://www.googleapis.com/auth/gmail.modify'
DEFAULT_USER_ID = 'me'
DEFAULT_SENDER = 'anonymous bot'
RECEIVER_MAIL = 'guy7145@gmail.com'
# AL_CONFIG_LABEL = 'AL_CONFIG'
AL_CONFIG_LABEL = 'Label_4883971462481667562'


# xing yun means 'lucky' in chinese
class XingYun:
    def __init__(self):
        self.service = XingYun.__init_service()
        self.open_threads = {}
        return

    @staticmethod
    def __init_service():
        """
        :return: gmail api service
        """
        store = file.Storage('token.json')
        creds = store.get()

        if not creds or creds.invalid:
            flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
            creds = tools.run_flow(flow, store)

        service = build('gmail', 'v1', http=creds.authorize(Http()))
        return service

    def __send_message(self, msg):
        try:
            msg = self.service.users().messages().send(userId=DEFAULT_USER_ID, body=msg).execute()
            print('message sent (Message Id: {})'.format(Message.get_id(msg)))
            return msg

        except HttpError as error:
            print('An error occurred: {}'.format(error))

    def __reply(self, thread_data, body):
        thread_id = Thread.get_id(thread_data)
        last_msg_id = Message.get_id(Thread.get_messages(thread_data)[-1])

        references = thread_id
        in_reply_to = last_msg_id

        # subject of the first message, to prevent too many "Re:"s...
        subject = "Re: " + Message.get_subject(Thread.get_messages(thread_data)[0])

        self.__send_message(Message.create_message(RECEIVER_MAIL,
                                                   subject,
                                                   body,
                                                   thread_id,
                                                   DEFAULT_SENDER,
                                                   references,
                                                   in_reply_to))
        return

    @staticmethod
    def __is_msg_safe(msg):
        return AL_CONFIG_LABEL in Message.get_labels(msg)

    def __check_mail(self):
        threads = self.service.users().threads().list(userId='me').execute().get('threads', [])
        new_threads = []
        for thread in threads:
            thread_data = self.service.users().threads().get(userId=DEFAULT_USER_ID, id=thread['id']).execute()

            last_msg = Thread.get_messages(thread_data)[-1]

            nb_msgs = len(Thread.get_messages(thread_data))
            if nb_msgs > 1 or Message.is_outgoing(last_msg):
                # messages that were read but not yet handled are moved to trash to be marked as read
                pass
            elif XingYun.__is_msg_safe(last_msg):  # block messages from other people
                print('new message!')
                new_threads.append(thread_data)
        return new_threads

    def check_requests(self):
        ids, subjects, configs = [], [], []
        for thread_data in self.__check_mail():
            thread_id = Thread.get_id(thread_data)
            last_msg = Thread.get_messages(thread_data)[-1]
            msg_text = Message.to_string(last_msg)
            subject = Message.get_subject(last_msg)
            try:
                conf = Configuration.from_json(msg_text)
            except ConfigurationParsingError as e:
                self.reply_config_error(thread_data, e)
                self.move_to_trash(thread_id)
                continue

            # if a configuration parsing error has been raised, the id won't be handled
            self.open_threads[thread_id] = thread_data
            ids.append(thread_id)
            subjects.append(subject)
            configs.append(conf)

        return ids, subjects, configs

    def reply_config_error(self, thread_data, error):
        self.__reply(thread_data, str(error))
        return

    def reply_config_ok(self, thread_id):
        self.__reply(self.open_threads[thread_id], "got it. getting to work")
        return

    def report_results(self, thread_id, results):
        self.move_to_trash(thread_id)

        thread_data = self.open_threads[thread_id]
        del self.open_threads[thread_id]
        self.__reply(thread_data, results)
        return

    def move_to_trash(self, thread_id):
        self.service.users().threads().trash(userId=DEFAULT_USER_ID, id=thread_id).execute()
        return

    def notify(self, subject, msg):
        self.__send_message(Message.create_message(RECEIVER_MAIL, subject, message_text=msg))
        return

    def try_notify(self, *args, **kwargs):
        try:
            self.notify(*args, **kwargs)
        except:
            print("couldn't notify by mail", file=stderr)
        return
