"""Storage for questions and answers in PostgreSQL."""

import logging
from typing import Dict, List, Any, Optional
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc, text  # Import desc
import os  # Import os
import openai  # Import openai

from dbt_llm_agent.core.models import Base, Question, QuestionTable, QuestionModelTable
from pgvector.sqlalchemy import Vector  # Import Vector

logger = logging.getLogger(__name__)

# Assuming OpenAI embedding model details
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIM = 1536


class QuestionStorage:
    """Storage service for questions and answers."""

    def __init__(self, connection_string: str, openai_api_key: Optional[str] = None):
        """Initialize the question storage service.

        Args:
            connection_string: SQLAlchemy connection string for PostgreSQL
            openai_api_key: OpenAI API key for embeddings. Reads from OPENAI_API_KEY env var if not provided.
        """
        self.engine = sa.create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)

        # Initialize OpenAI client
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self._openai_api_key:
            logger.warning(
                "OpenAI API key not provided. Question embedding generation will be skipped."
            )
            self.openai_client = None
        else:
            try:
                self.openai_client = openai.OpenAI(api_key=self._openai_api_key)
                logger.info("OpenAI client initialized for question embeddings.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                self.openai_client = None

        # Create tables if they don't exist
        self._create_tables()

        logger.info("Initialized question storage service")

    def _create_tables(self):
        """Create the necessary tables if they don't exist."""
        # Enable pgvector extension
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        Base.metadata.create_all(self.engine)
        logger.info("Created question tracking tables (ensured vector extension)")

    def _get_embedding(self, text_to_embed: str) -> Optional[List[float]]:
        """Generate embedding for the given text using OpenAI.

        Args:
            text_to_embed: The text to embed.

        Returns:
            The embedding vector, or None if client is not available or error occurs.
        """
        if not self.openai_client:
            logger.warning("OpenAI client not available, cannot generate embedding.")
            return None
        try:
            response = self.openai_client.embeddings.create(
                model=EMBEDDING_MODEL, input=[text_to_embed]
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding for text: {e}")
            return None

    def record_question(
        self,
        question_text: str,
        answer_text: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        was_useful: Optional[bool] = None,
        feedback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a question and its answer in the database, including its embedding.

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
            question_orm = domain_question.to_orm()

            # Generate and add embedding
            embedding = self._get_embedding(question_text)
            if embedding:
                question_orm.question_embedding = embedding
            else:
                logger.warning(
                    f"Could not generate embedding for question: {question_text[:50]}..."
                )

            session.add(question_orm)
            session.flush()  # Get the assigned ID
            question_id = question_orm.id

            # Add associated models if provided
            if model_names:
                for model_name in model_names:
                    question_model = QuestionModelTable(
                        question_id=question_id,
                        model_name=model_name,
                    )
                    session.add(question_model)

            # Commit changes
            session.commit()
            logger.info(f"Recorded question with ID {question_id}")
            return question_id
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

    def find_similar_questions_with_feedback(
        self,
        query_embedding: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.8,  # Cosine similarity threshold
    ) -> List[Question]:
        """Find similar questions that have feedback (marked not useful or have feedback text).

        Args:
            query_embedding: The embedding vector of the current question.
            limit: Max number of similar questions to return.
            similarity_threshold: Minimum cosine similarity for a question to be considered.

        Returns:
            A list of Question domain objects, ordered by similarity and recency.
        """
        session = self.Session()
        distance_threshold = (
            1 - similarity_threshold
        )  # Convert cosine similarity to cosine distance
        try:
            similar_questions = (
                session.query(QuestionTable)
                .filter(
                    # Consider questions with any feedback (text or was_useful flag set)
                    sa.or_(
                        QuestionTable.feedback.isnot(None),
                        QuestionTable.was_useful.isnot(
                            None
                        ),  # Check if was_useful is set (True or False)
                    )
                )
                .filter(
                    QuestionTable.question_embedding.isnot(None)
                )  # Ensure embedding exists
                .filter(
                    QuestionTable.question_embedding.cosine_distance(query_embedding)
                    < distance_threshold
                )
                .order_by(
                    QuestionTable.question_embedding.cosine_distance(
                        query_embedding
                    ).asc()
                )
                .order_by(
                    QuestionTable.updated_at.desc()
                )  # Prioritize more recent feedback for ties
                .limit(limit)
                .all()
            )

            # Convert ORM results to domain models
            return [q.to_domain() for q in similar_questions]
        except Exception as e:
            logger.error(f"Error finding similar questions: {e}")
            return []
        finally:
            session.close()
