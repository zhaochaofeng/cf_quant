'''
    Prefect 相关代码
'''

from .utils import get_config
from prefect_email import EmailServerCredentials
from prefect import flow
from prefect_email import email_send_message
from typing import Union, List, Optional


def register_smtp(name):
    """ 注册SMTP
        name: 注册发送邮件的SMTP名称
        如: register_smtp('alert')
    """
    config = get_config()
    credentials = EmailServerCredentials(
        username=config['email']['sender'],
        password=config['email']['password'],
        smtp_server=config['email']['smtpserver'],
        smtp_port=config['email']['port'],
    )
    credentials.save(name)

@flow
def email_send_message_flow(email_addresses: Optional[Union[List[str], str]] = None, subject: str = None, msg: str = None):
    """ 发送邮件
        email_send_message_flow('469293319@qq.com', 'test', 'send message')
    """
    if email_addresses is None:
        config = get_config()
        email_addresses = config['email']['receiver']
        print(email_addresses)
    if isinstance(email_addresses, str):
        email_addresses = [email_addresses]
    email_server_credentials = EmailServerCredentials.load("alert")
    for email_address in email_addresses:
        subject = email_send_message.with_options(name=f"email {email_address}").submit(
            email_server_credentials=email_server_credentials,
            subject=subject,
            msg=msg,
            email_to=email_address,
        )
