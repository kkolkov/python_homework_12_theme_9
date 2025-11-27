from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, select
from sqlalchemy.orm import declarative_base, sessionmaker

import pandas as pd
import secrets
import redis
import json

Base = declarative_base()

# =====================================================
#                     REDIS
# =====================================================
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def cache_get(key: str):
    data = redis_client.get(key)
    if data:
        return json.loads(data)
    return None


def cache_set(key: str, value, ttl=30):
    redis_client.set(key, json.dumps(value), ex=ttl)


# =====================================================
#                     МОДЕЛИ
# =====================================================
class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    last_name = Column(String, nullable=False)
    first_name = Column(String, nullable=False)
    faculty = Column(String, nullable=False)
    course = Column(String, nullable=False)
    grade = Column(Float, nullable=False)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True)
    password = Column(String)
    token = Column(String, nullable=True)
    role = Column(String, default="write")  # write / read


# =====================================================
#                     SCHEMAS
# =====================================================
class StudentCreate(BaseModel):
    last_name: str
    first_name: str
    faculty: str
    course: str
    grade: float


class StudentUpdate(BaseModel):
    last_name: Optional[str] = None
    first_name: Optional[str] = None
    faculty: Optional[str] = None
    course: Optional[str] = None
    grade: Optional[float] = None


class StudentOut(BaseModel):
    id: int
    last_name: str
    first_name: str
    faculty: str
    course: str
    grade: float

    class Config:
        orm_mode = True


class UserCreate(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


# =====================================================
#                  КЛАСС РАБОТЫ С БД
# =====================================================
class Database:
    def __init__(self, db_url="sqlite:///students.db"):
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    # -------- USERS --------
    def create_user(self, username, password):
        with self.Session() as session:
            exists = session.execute(
                select(User).where(User.username == username)
            ).scalar_one_or_none()
            if exists:
                raise HTTPException(400, "User already exists")

            user = User(username=username, password=password)
            session.add(user)
            session.commit()
            return {"status": "registered"}

    def login(self, username, password):
        with self.Session() as session:
            user = session.execute(
                select(User).where(User.username == username)
            ).scalar_one_or_none()

            if not user or user.password != password:
                raise HTTPException(401, "Invalid credentials")

            token = secrets.token_hex(16)
            user.token = token
            session.commit()

            return {"token": token}

    def logout(self, token):
        with self.Session() as session:
            user = session.execute(
                select(User).where(User.token == token)
            ).scalar_one_or_none()
            if not user:
                raise HTTPException(401, "Invalid token")

            user.token = None
            session.commit()
            return {"status": "logged_out"}

    def check_token(self, token):
        with self.Session() as session:
            return session.execute(
                select(User).where(User.token == token)
            ).scalar_one_or_none()

    # -------- STUDENTS Methods --------
    def create_student(self, data: StudentCreate):
        with self.Session() as session:
            student = Student(**data.dict())
            session.add(student)
            session.commit()
            session.refresh(student)
            return student

    def get_all_students(self):
        with self.Session() as session:
            return session.scalars(select(Student)).all()

    def get_student(self, student_id: int):
        with self.Session() as session:
            return session.get(Student, student_id)

    def update_student(self, student_id: int, data: StudentUpdate):
        with self.Session() as session:
            student = session.get(Student, student_id)
            if not student:
                return None
            for key, value in data.dict(exclude_unset=True).items():
                setattr(student, key, value)
            session.commit()
            session.refresh(student)
            return student

    def delete_student(self, student_id: int):
        with self.Session() as session:
            student = session.get(Student, student_id)
            if not student:
                return False
            session.delete(student)
            session.commit()
            return True


db = Database()

# =====================================================
#                ФОНОВЫЕ ЗАДАЧИ
# =====================================================

def load_students_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    with db.Session() as session:
        for _, row in df.iterrows():
            student = Student(
                last_name=row["last_name"],
                first_name=row["first_name"],
                faculty=row["faculty"],
                course=row["course"],
                grade=row["grade"]
            )
            session.add(student)
        session.commit()

    redis_client.flushdb()


def delete_students_list(ids: List[int]):
    with db.Session() as session:
        for student_id in ids:
            student = session.get(Student, student_id)
            if student:
                session.delete(student)
        session.commit()

    redis_client.flushdb()


# =====================================================
#                FASTAPI + AUTH
# =====================================================
app = FastAPI(title="Students API + Auth")


# ------------------ AUTH DEPENDENCY ------------------
def require_user(Authorization: str = Header(None)):
    if not Authorization or not Authorization.startswith("Token "):
        raise HTTPException(401, "No authentication token")

    token = Authorization.split()[1]
    user = db.check_token(token)
    if not user:
        raise HTTPException(401, "Invalid token")

    return user


# ------------------ AUTH ROUTES ------------------
@app.post("/auth/register")
def register(user: UserCreate):
    return db.create_user(user.username, user.password)


@app.post("/auth/login")
def login(data: UserLogin):
    return db.login(data.username, data.password)


@app.post("/auth/logout")
def logout(user=Depends(require_user)):
    return db.logout(user.token)


# =====================================================
#                     ФОНОВЫЕ ЭНДПОЙНТЫ
# =====================================================

@app.post("/background/load_csv")
def load_csv(csv_path: str, background: BackgroundTasks, user=Depends(require_user)):
    if user.role != "write":
        raise HTTPException(403, "Read-only user")

    background.add_task(load_students_from_csv, csv_path)
    return {"status": "CSV loading started"}


@app.post("/background/delete_list")
def delete_list(ids: List[int], background: BackgroundTasks, user=Depends(require_user)):
    if user.role != "write":
        raise HTTPException(403, "Read-only user")

    background.add_task(delete_students_list, ids)
    return {"status": "Delete task started"}


# =====================================================
#                     CRUD (с кешированием)
# =====================================================

@app.post("/students/", response_model=StudentOut)
def create_student(student: StudentCreate, user=Depends(require_user)):
    if user.role != "write":
        raise HTTPException(403, "Read-only user")

    redis_client.flushdb()
    return db.create_student(student)


@app.get("/students/", response_model=List[StudentOut])
def get_students(user=Depends(require_user)):
    cache_key = "students_all"
    cached = cache_get(cache_key)
    if cached:
        return cached

    data = db.get_all_students()
    data_out = [StudentOut.from_orm(s).dict() for s in data]

    cache_set(cache_key, data_out)
    return data_out


@app.get("/students/{student_id}", response_model=StudentOut)
def get_student(student_id: int, user=Depends(require_user)):
    cache_key = f"student_{student_id}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    student = db.get_student(student_id)
    if not student:
        raise HTTPException(404, "Student not found")

    data_out = StudentOut.from_orm(student).dict()
    cache_set(cache_key, data_out)
    return data_out


@app.put("/students/{student_id}", response_model=StudentOut)
def update_student(student_id: int, data: StudentUpdate, user=Depends(require_user)):
    if user.role != "write":
        raise HTTPException(403, "Read-only user")

    student = db.update_student(student_id, data)
    if not student:
        raise HTTPException(404, "Student not found")

    redis_client.flushdb()
    return student


@app.delete("/students/{student_id}")
def delete_student(student_id: int, user=Depends(require_user)):
    if user.role != "write":
        raise HTTPException(403, "Read-only user")

    ok = db.delete_student(student_id)
    if not ok:
        raise HTTPException(404, "Student not found")

    redis_client.flushdb()
    return {"status": "deleted"}
