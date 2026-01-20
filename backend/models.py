from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TruthImage(Base):
    __tablename__ = "truth_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_path: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[str] = mapped_column(Text, nullable=False)  # JSON list[float]
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def embedding_as_list(self) -> list[float]:
        return json.loads(self.embedding)


class TestImage(Base):
    __tablename__ = "test_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_path: Mapped[str] = mapped_column(String, nullable=False)
    match_status: Mapped[str] = mapped_column(String, nullable=False)  # MATCH / NO_MATCH / NO_FACE / NO_TRUTH
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TestVideo(Base):
    __tablename__ = "test_videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_path: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    matches: Mapped[list["VideoMatch"]] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class VideoMatch(Base):
    __tablename__ = "video_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("test_videos.id"), nullable=False)
    frame_path: Mapped[str] = mapped_column(String, nullable=False)
    match_status: Mapped[str] = mapped_column(String, nullable=False)  # MATCH / NO_MATCH / NO_FACE / NO_TRUTH
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    video: Mapped["TestVideo"] = relationship(back_populates="matches")


