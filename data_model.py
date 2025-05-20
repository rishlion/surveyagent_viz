from typing import Optional
from sqlmodel import SQLModel, Field, create_engine, Session

class SurveyResponse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    respondent_id: str
    age: Optional[int]
    gender: Optional[str]
    region: Optional[str]
    question: str
    answer: str
    confidence: float

def _engine(db_path: str):
    return create_engine(f"sqlite:///{db_path}", echo=False)

def create_db_and_tables(db_path: str):
    SQLModel.metadata.create_all(_engine(db_path))

def get_session(db_path: str) -> Session:
    return Session(_engine(db_path))

def add_response(session: Session, record, question: str, answer: str, confidence: float):
    resp = SurveyResponse(
        respondent_id=record["respondent_id"],
        age=record.get("age"),
        gender=record.get("gender"),
        region=record.get("region"),
        question=question,
        answer=answer,
        confidence=confidence,
    )
    session.add(resp)
    session.commit()
