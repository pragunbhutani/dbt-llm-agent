"""Storage for questions and answers in PostgreSQL."""

import logging
from typing import Dict, List, Any, Optional
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy import desc, text  # Import desc
import os  # Import os
import openai  # Import openai

from ragstar_archive.core.models import (
    Base,
    Question,
    QuestionTable,
    QuestionModelTable,
)
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
        # This will create all tables defined in Base metadata, including the new column
        Base.metadata.create_all(self.engine)
        logger.info(
            "Created/verified question tracking tables (ensured vector extension)"
        )

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
        original_message_text: Optional[str] = None,
        original_message_ts: Optional[str] = None,
        response_message_ts: Optional[str] = None,
        response_file_message_ts: Optional[str] = None,
        answer_text: Optional[str] = None,
        model_names: Optional[List[str]] = None,
        was_useful: Optional[bool] = None,
        feedback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Record a question and its answer, including embeddings.

        Args:
            question_text: The text of the (potentially rephrased) question.
            original_message_text: The original verbatim text from the user's message.
            original_message_ts: The timestamp (ID) of the original user message.
            response_message_ts: The timestamp (ID) of the bot's final response message.
            response_file_message_ts: The timestamp (ID) of the message containing the uploaded file snippet.
            answer_text: The text of the answer generated.
            model_names: Names of models used to answer the question.
            was_useful: Whether the answer was marked useful.
            feedback: User feedback text.
            metadata: Additional metadata.

        Returns:
            The ID of the recorded question.
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
                original_message_text=original_message_text,
                original_message_ts=original_message_ts,
                response_message_ts=response_message_ts,
                response_file_message_ts=response_file_message_ts,
            )

            # Convert to ORM model
            question_orm = (
                domain_question.to_orm()
            )  # This now includes None for embeddings initially

            # Generate and add question embedding (for the compiled question_text)
            question_embedding = self._get_embedding(question_text)
            if question_embedding:
                question_orm.question_embedding = question_embedding
            else:
                logger.warning(
                    f"Could not generate embedding for question_text: {question_text[:50]}..."
                )

            # Generate and add original message embedding
            if original_message_text:
                original_embedding = self._get_embedding(original_message_text)
                if original_embedding:
                    question_orm.original_message_embedding = original_embedding
                else:
                    logger.warning(
                        f"Could not generate embedding for original_message_text: {original_message_text[:50]}..."
                    )

            # Generate and add feedback embedding if feedback text exists
            if feedback:
                feedback_embedding = self._get_embedding(feedback)
                if feedback_embedding:
                    question_orm.feedback_embedding = feedback_embedding
                else:
                    logger.warning(
                        f"Could not generate embedding for feedback: {feedback[:50]}..."
                    )

            session.add(question_orm)
            session.flush()  # Get the assigned ID
            question_id = question_orm.id

            # Add associated models if provided
            if model_names:
                for model_name in model_names:
                    # Assuming QuestionModelTable takes question_id and model_name
                    question_model = QuestionModelTable(
                        question_id=question_id,
                        model_name=model_name,
                        # relevance_score might be needed if your model includes it
                    )
                    session.add(question_model)

            session.commit()
            logger.info(f"Recorded question with ID {question_id}")
            return question_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording question: {e}", exc_info=True)
            raise
        finally:
            session.close()

    def update_feedback(
        self,
        item_identifier: str,
        item_type: str,
        was_useful: Optional[bool],
        feedback_provider_user_id: str,
    ) -> bool:
        """Finds a question by its response message timestamp OR response file message timestamp and updates feedback.

        Args:
            item_identifier: The timestamp ('ts') of the Slack message reacted to.
            item_type: Must be 'message' (identifies either the text response or the file upload message).
            was_useful: True for positive feedback, False for negative, None to remove feedback.
            feedback_provider_user_id: The Slack user ID of the user who provided the feedback.

        Returns:
            True if the corresponding question record was found and updated, False otherwise.
        """
        session = self.Session()
        try:
            query = session.query(QuestionTable)

            if item_type == "message":
                # Find if the reaction timestamp matches either the text response OR the file response message timestamp
                query = query.filter(
                    sa.or_(
                        QuestionTable.response_message_ts == item_identifier,
                        QuestionTable.response_file_message_ts == item_identifier,
                    )
                )
            # REMOVED: elif item_type == "file": block, as reactions now always come as 'message' type
            else:
                logger.warning(
                    f"Invalid item_type '{item_type}' for feedback update. Expected 'message'."
                )
                return False

            question = query.first()

            if not question:
                logger.warning(
                    f"No question found associated with message identifier: {item_identifier}"
                )
                return False

            question.was_useful = was_useful

            session.commit()
            logger.info(
                f"Updated feedback (was_useful={was_useful}) for question associated with message {item_identifier}"
            )
            return True
        except Exception as e:
            session.rollback()
            logger.error(
                f"Error updating feedback by message {item_identifier}: {e}",
                exc_info=True,
            )
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
        similarity_threshold: float = 0.65,  # Cosine similarity threshold for question_text
    ) -> List[Question]:
        """Find similar questions (based on compiled question_text) that have feedback.

        Args:
            query_embedding: Embedding vector of the *current* question's compiled text.
            limit: Max number of similar questions to return.
            similarity_threshold: Minimum cosine similarity for question_text.

        Returns:
            List of Question domain objects, ordered by similarity.
        """
        session = self.Session()
        distance_threshold = 1 - similarity_threshold
        try:
            similar_questions = (
                session.query(QuestionTable)
                .filter(
                    sa.or_(
                        QuestionTable.feedback.isnot(None),
                        QuestionTable.was_useful.isnot(None),
                    )
                )
                .filter(QuestionTable.question_embedding.isnot(None))
                .filter(
                    QuestionTable.question_embedding.cosine_distance(query_embedding)
                    < distance_threshold  # Use cosine distance
                )
                .order_by(
                    QuestionTable.question_embedding.cosine_distance(
                        query_embedding
                    ).asc()
                )
                .limit(limit)
                .all()
            )
            return [q.to_domain() for q in similar_questions]
        except Exception as e:
            logger.error(f"Error finding similar questions with feedback: {e}")
            return []
        finally:
            session.close()

    def find_similar_feedback_content(
        self,
        query: str,
        limit: int = 3,
        similarity_threshold: float = 0.6,  # Cosine similarity threshold for feedback content
    ) -> List[Question]:
        """Find questions where the *feedback text* is semantically similar to the query.

        Args:
            query: The text query to search for within feedback.
            limit: Max number of matching questions to return.
            similarity_threshold: Minimum cosine similarity for feedback text.

        Returns:
            List of Question domain objects whose feedback is relevant.
        """
        session = self.Session()
        query_embedding = self._get_embedding(query)
        if not query_embedding:
            logger.warning(
                f"Could not generate embedding for feedback content query: {query[:50]}..."
            )
            return []

        distance_threshold = 1 - similarity_threshold
        try:
            similar_feedback_questions = (
                session.query(QuestionTable)
                .filter(
                    QuestionTable.feedback.isnot(None),
                    QuestionTable.feedback_embedding.isnot(None),
                )
                .filter(
                    QuestionTable.feedback_embedding.cosine_distance(query_embedding)
                    < distance_threshold  # Use cosine distance
                )
                .order_by(
                    QuestionTable.feedback_embedding.cosine_distance(
                        query_embedding
                    ).asc()
                )
                .limit(limit)
                .all()
            )
            return [q.to_domain() for q in similar_feedback_questions]
        except Exception as e:
            logger.error(f"Error finding similar feedback content: {e}")
            return []
        finally:
            session.close()

    # --- NEW METHOD: Search original message content --- #
    def find_similar_original_messages(
        self,
        query_embedding: List[float],
        limit: int = 3,
        similarity_threshold: float = 0.7,  # Cosine similarity threshold for original message
    ) -> List[Question]:
        """Find past interactions where the *original user message text* is semantically similar to the query embedding.

        This is useful for finding organizational context, definitions, or explanations
        provided in past questions, distinct from searching compiled questions or feedback.

        Args:
            query_embedding: The embedding vector of the *current* original user message.
            limit: Max number of similar original messages to return.
            similarity_threshold: Minimum cosine similarity for original_message_text.

        Returns:
            A list of Question domain objects whose original message is relevant,
            ordered by similarity.
        """
        session = self.Session()
        # For cosine similarity 's', distance 'd' is 1 - s. We want distance < (1 - similarity_threshold)
        distance_threshold = 1 - similarity_threshold

        try:
            similar_original_message_questions = (
                session.query(QuestionTable)
                .filter(
                    QuestionTable.original_message_text.isnot(None),
                    QuestionTable.original_message_embedding.isnot(None),
                )
                .filter(
                    QuestionTable.original_message_embedding.cosine_distance(
                        query_embedding
                    )
                    < distance_threshold  # Use cosine distance
                )
                .order_by(
                    QuestionTable.original_message_embedding.cosine_distance(
                        query_embedding
                    ).asc()
                )
                # Optional: Add secondary sort by recency if needed
                .order_by(QuestionTable.created_at.desc())
                .limit(limit)
                .all()
            )

            # Convert ORM results to domain models
            return [q.to_domain() for q in similar_original_message_questions]
        except Exception as e:
            logger.error(
                f"Error finding similar original message content: {e}", exc_info=True
            )
            return []
        finally:
            session.close()

    # --- END NEW METHOD --- #
