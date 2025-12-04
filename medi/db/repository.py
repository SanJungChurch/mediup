# db/repository.py
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트에 'wellness.db'라는 파일로 저장됩니다
DB_PATH = Path(__file__).parent.parent / "wellness.db"

class LogRepository:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # 원하시는 9개 항목 + 시간(ts) 저장용 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                perclos REAL,
                yawn_rate REAL,
                posture_angle REAL,
                headpose_var REAL,
                fatigue REAL,
                stress REAL,
                blink INTEGER,
                yawn INTEGER,
                nodding INTEGER
            )
        ''')
        conn.commit()
        conn.close()

    def save(self, data: dict):
        """n초마다 호출될 저장 함수"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO logs (
                ts, perclos, yawn_rate, posture_angle, headpose_var,
                fatigue, stress, blink, yawn, nodding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            data['perclos'],
            data['yawn_rate'],
            data['posture'],
            data['headpose'],
            data['fatigue'],
            data['stress'],

            1 if data['blink'] else 0, # True/False -> 1/0 변환
            1 if data['yawn'] else 0,
            1 if data['nodding'] else 0
        ))
        conn.commit()
        conn.close()

        # db/repository.py (기존 클래스 안에 메서드 추가)
    
    def get_data_for_analysis(self, hours: int = 24):
        """최근 N시간 동안의 6가지 지표 데이터를 모두 가져옴"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 현재 시간 - hours
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        # 6가지 핵심 지표만 조회
        cursor.execute('''
            SELECT ts, perclos, yawn_rate, posture_angle, headpose_var, fatigue, stress
            FROM logs 
            WHERE ts > ? 
            ORDER BY ts ASC
        ''', (cutoff_time,))
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

# 전역 객체 생성
repo = LogRepository()