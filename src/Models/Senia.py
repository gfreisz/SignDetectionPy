from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker

eng = create_engine('mysql://root:sorditos1@localhost/peals')
Base = declarative_base()
metadata = MetaData(bind=eng)

class Senia(Base):
    __table__ = Table('Senia', metadata, autoload=True)

    def __init__(self):
        Session = sessionmaker(bind=eng)
        self.__ses = Session()

    def query(self):
        return self.__ses.query(Senia)#.all()
