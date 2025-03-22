"""Service for tracking questions and answers."""

import logging
from typing import Dict, List, Any, Optional
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from dbt_llm_agent.storage.models import Base, Question, QuestionModel

logger = logging.getLogger(__name__)


class QuestionTrackingService:
    """Service for tracking questions and answers."""

    def __init__(self, connection_string: str):
        """Initialize the question tracking service.

        Args:
            connection_string: SQLAlchemy connection string for PostgreSQL
        """
        self.engine = sa.create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        self._create_tables()

        logger.info("Initialized question tracking service")

    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        logger.info("Created question tracking tables")

    def record_question(
        self,
        question_text: str,
        answer_text: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        was_useful: Optional[bool] = None,
        feedback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a question and its answer in the database.

        Args:
            question_text: The text of the question
            answer_text: The text of the answer
            model_names: Names of models that were used to answer the question
            was_useful: Whether the answer was useful
            feedback: User feedback on the answer
            metadata: Additional metadata

        Returns:
            The ID of the recorded question
        """
        session = self.Session()
        try:
            # Create question
            question = Question(
                question_text=question_text,
                answer_text=answer_text,
                was_useful=was_useful,
                feedback=feedback,
                question_metadata=metadata or {},
            )
            session.add(question)

            # Add associated models if provided
            if model_names:
                for model_name in model_names:
                    question_model = QuestionModel(
                        model_name=model_name,
                    )
                    question.models.append(question_model)

            # Commit changes
            session.commit()
            logger.info(f"Recorded question with ID {question.id}")
            return question.id
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording question: {e}")
            raise
        finally:
            session.close()

    def update_feedback(
        self, question_id: int, was_useful: bool, feedback: Optional[str] = None
    ) -> bool:
        """Update feedback for a question.

        Args:
            question_id: The ID of the question
            was_useful: Whether the answer was useful
            feedback: User feedback on the answer

        Returns:
            Whether the update was successful
        """
        session = self.Session()
        try:
            # Get question
            question = (
                session.query(Question).filter(Question.id == question_id).first()
            )
            if not question:
                logger.warning(f"Question with ID {question_id} not found")
                return False

            # Update feedback
            question.was_useful = was_useful
            if feedback:
                question.feedback = feedback

            # Commit changes
            session.commit()
            logger.info(f"Updated feedback for question with ID {question_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating feedback: {e}")
            return False
        finally:
            session.close()

    def get_question(self, question_id: int) -> Optional[Dict[str, Any]]:
        """Get a question by ID.

        Args:
            question_id: The ID of the question

        Returns:
            The question data as a dictionary, or None if not found
        """
        session = self.Session()
        try:
            # Get question
            question = (
                session.query(Question).filter(Question.id == question_id).first()
            )
            if not question:
                logger.warning(f"Question with ID {question_id} not found")
                return None

            # Get associated models
            question_models = (
                session.query(QuestionModel)
                .filter(QuestionModel.question_id == question_id)
                .all()
            )

            # Format response
            return {
                "id": question.id,
                "question_text": question.question_text,
                "answer_text": question.answer_text,
                "was_useful": question.was_useful,
                "feedback": question.feedback,
                "metadata": question.question_metadata,
                "created_at": (
                    question.created_at.isoformat() if question.created_at else None
                ),
                "updated_at": (
                    question.updated_at.isoformat() if question.updated_at else None
                ),
                "models": [
                    {
                        "name": qm.model_name,
                        "relevance_score": qm.relevance_score,
                    }
                    for qm in question_models
                ],
            }
        except Exception as e:
            logger.error(f"Error getting question: {e}")
            return None
        finally:
            session.close()

    def get_all_questions(
        self, limit: int = 100, offset: int = 0, was_useful: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Get all questions.

        Args:
            limit: Maximum number of questions to return
            offset: Offset for pagination
            was_useful: Filter by usefulness

        Returns:
            A list of questions
        """
        session = self.Session()
        try:
            # Build query
            query = session.query(Question)

            # Apply filter if provided
            if was_useful is not None:
                query = query.filter(Question.was_useful == was_useful)

            # Apply pagination
            questions = (
                query.order_by(Question.created_at.desc())
                .limit(limit)
                .offset(offset)
                .all()
            )

            # Format response
            return [
                {
                    "id": q.id,
                    "question_text": q.question_text,
                    "answer_text": q.answer_text,
                    "was_useful": q.was_useful,
                    "feedback": q.feedback,
                    "created_at": q.created_at.isoformat() if q.created_at else None,
                    "models": [
                        {
                            "name": qm.model_name,
                            "relevance_score": qm.relevance_score,
                        }
                        for qm in q.models
                    ],
                }
                for q in questions
            ]
        except Exception as e:
            logger.error(f"Error getting questions: {e}")
            return []
        finally:
            session.close()
