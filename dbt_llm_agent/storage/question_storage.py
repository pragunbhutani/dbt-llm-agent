"""Storage for questions and answers in PostgreSQL."""

import logging
from typing import Dict, List, Any, Optional
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from dbt_llm_agent.core.models import Base, Question, QuestionTable, QuestionModelTable

logger = logging.getLogger(__name__)


class QuestionStorage:
    """Storage service for questions and answers."""

    def __init__(self, connection_string: str):
        """Initialize the question storage service.

        Args:
            connection_string: SQLAlchemy connection string for PostgreSQL
        """
        self.engine = sa.create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)

        # Create tables if they don't exist
        self._create_tables()

        logger.info("Initialized question storage service")

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
            # Create a domain model first
            domain_question = Question(
                question_text=question_text,
                answer_text=answer_text,
                was_useful=was_useful,
                feedback=feedback,
                question_metadata=metadata or {},
            )

            # Convert to ORM model
            question = domain_question.to_orm()
            session.add(question)
            session.flush()  # Get the assigned ID

            # Add associated models if provided
            if model_names:
                for model_name in model_names:
                    question_model = QuestionModelTable(
                        question_id=question.id,
                        model_name=model_name,
                    )
                    session.add(question_model)

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
                session.query(QuestionTable)
                .filter(QuestionTable.id == question_id)
                .first()
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

    def get_question(self, question_id: int) -> Optional[Question]:
        """Get a question by ID.

        Args:
            question_id: The ID of the question

        Returns:
            The domain Question object, or None if not found
        """
        session = self.Session()
        try:
            # Get question
            question = (
                session.query(QuestionTable)
                .filter(QuestionTable.id == question_id)
                .first()
            )
            if not question:
                logger.warning(f"Question with ID {question_id} not found")
                return None

            # Convert to domain model using to_domain method
            return question.to_domain()
        except Exception as e:
            logger.error(f"Error getting question: {e}")
            return None
        finally:
            session.close()

    def get_all_questions(
        self, limit: int = 100, offset: int = 0, was_useful: Optional[bool] = None
    ) -> List[Question]:
        """Get all questions.

        Args:
            limit: Maximum number of questions to return
            offset: Offset for pagination
            was_useful: Filter by usefulness

        Returns:
            A list of domain Question objects
        """
        session = self.Session()
        try:
            # Build query
            query = session.query(QuestionTable)

            # Apply filter if provided
            if was_useful is not None:
                query = query.filter(QuestionTable.was_useful == was_useful)

            # Apply pagination
            questions = (
                query.order_by(QuestionTable.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            # Convert to domain models
            return [q.to_domain() for q in questions]
        except Exception as e:
            logger.error(f"Error getting questions: {e}")
            return []
        finally:
            session.close()
