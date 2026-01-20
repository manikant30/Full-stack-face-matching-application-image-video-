from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Flask, jsonify
from flask import send_from_directory
from flask_cors import CORS
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import Config
from models import Base
from routes.test_images import test_images_bp
from routes.test_videos import test_videos_bp
from routes.truth_image import truth_image_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    # Basic, helpful logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # CORS so React can call Flask.
    CORS(app, resources={r"/api/*": {"origins": app.config["FRONTEND_ORIGIN"]}})

    # Ensure uploads folders exist.
    uploads_dir = Path(app.config["UPLOAD_FOLDER"])
    (uploads_dir / "truth").mkdir(parents=True, exist_ok=True)
    (uploads_dir / "images").mkdir(parents=True, exist_ok=True)
    (uploads_dir / "videos").mkdir(parents=True, exist_ok=True)
    (uploads_dir / "frames").mkdir(parents=True, exist_ok=True)

    # SQLAlchemy (plain, no Flask-SQLAlchemy to keep it simple).
    db_url = app.config["DATABASE_URL"]
    connect_args = {}
    # Improve SQLite dev experience: allow reuse across threads and increase lock timeout.
    if str(db_url).startswith("sqlite"):
        connect_args = {"check_same_thread": False, "timeout": 30}

    engine = create_engine(db_url, echo=False, connect_args=connect_args)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    # Put on app for easy access in routes.
    app.session_local = SessionLocal  # type: ignore[attr-defined]

    # Register routes.
    app.register_blueprint(truth_image_bp, url_prefix="/api")
    app.register_blueprint(test_images_bp, url_prefix="/api")
    app.register_blueprint(test_videos_bp, url_prefix="/api")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/uploads/<path:filename>")
    def serve_upload(filename: str):
        """
        Serve uploaded files for previews (dev-friendly).
        Frontend can request: http://localhost:5000/uploads/<subpath>
        """
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)

    @app.errorhandler(Exception)
    def handle_exception(e: Exception):
        # Keep errors JSON for frontend.
        logging.exception("Unhandled error: %s", e)
        return jsonify({"error": "Server error", "detail": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", "5000"))
    # IMPORTANT (Windows/dev): uploads create new files in backend/uploads/.
    # If the Werkzeug reloader is enabled, it detects file changes and restarts
    # the server mid-upload -> CRA proxy shows ECONNRESET.
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)


