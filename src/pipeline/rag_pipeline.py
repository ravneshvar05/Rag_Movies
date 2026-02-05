"""
Main RAG pipeline orchestrating all components.
"""
import time
from typing import Optional
from src.models.schemas import RAGResult, QuestionRoute, Answer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.router import QuestionRouter
from src.llm.relevance_judge import RelevanceJudge
from src.llm.distiller import ContextDistiller
from src.llm.answerer import Answerer
from src.utils.logger import MovieRAGLogger

logger = MovieRAGLogger.get_logger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline for movie transcript QA.
    """
    
    def __init__(
        self,
        retriever: HybridRetriever,
        router: QuestionRouter,
        judge: RelevanceJudge,
        distiller: ContextDistiller,
        answerer: Answerer
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: HybridRetriever instance
            router: QuestionRouter instance
            judge: RelevanceJudge instance
            distiller: ContextDistiller instance
            answerer: Answerer instance
        """
        self.retriever = retriever
        self.router = router
        self.judge = judge
        self.distiller = distiller
        self.answerer = answerer
        self.logger = logger
    
    def process(
        self,
        question: str,
        movie_id: Optional[str] = None
    ) -> RAGResult:
        """
        Process a question through the complete RAG pipeline.
        
        Args:
            question: User question
            movie_id: Optional movie identifier to filter results
            
        Returns:
            RAGResult with answer and metadata
        """
        start_time = time.time()
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Processing question: {question}")
        self.logger.info(f"{'='*80}")
        
        try:
            # Step 1: Route question
            self.logger.info("\n[STEP 1] Routing question...")
            route = self.router.route(question)
            self.logger.info(f"Category: {route.category}")
            self.logger.info(f"Requires full narrative: {route.requires_full_narrative}")
            
            # Step 2: Retrieve relevant chunks
            self.logger.info("\n[STEP 2] Retrieving chunks...")
            retrieved_chunks = self.retriever.retrieve(
                query=question,
                movie_id=movie_id
            )
            self.logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            
            if not retrieved_chunks:
                self.logger.warning("No chunks retrieved")
                return self._create_no_context_result(question, route, start_time)
            
            # Step 3: Judge relevance
            self.logger.info("\n[STEP 3] Judging relevance...")
            judgment = self.judge.judge(question, retrieved_chunks)
            
            # Filter to relevant chunks
            relevant_chunk_ids = set(judgment.relevant_chunk_ids)
            relevant_chunks = [
                c for c in retrieved_chunks
                if c.chunk_id in relevant_chunk_ids
            ]
            
            self.logger.info(
                f"Filtered to {len(relevant_chunks)} relevant chunks "
                f"(from {len(retrieved_chunks)})"
            )
            
            if not relevant_chunks:
                self.logger.warning("No relevant chunks after filtering")
                return self._create_no_context_result(question, route, start_time)
            
            # Step 4: Distill context
            self.logger.info("\n[STEP 4] Distilling context...")
            distilled_context = self.distiller.distill(question, relevant_chunks)
            self.logger.info(
                f"Distilled {distilled_context.original_chunk_count} chunks "
                f"into {len(distilled_context.distilled_text)} characters"
            )
            
            # Step 5: Generate answer
            self.logger.info("\n[STEP 5] Generating answer...")
            answer = self.answerer.answer(question, distilled_context)
            self.logger.info(f"Answer generated with model: {answer.model_used}")
            self.logger.info(f"Confidence: {answer.confidence}")
            
            # Create result
            processing_time = time.time() - start_time
            
            result = RAGResult(
                question=question,
                route=route,
                retrieved_chunks=len(retrieved_chunks),
                relevant_chunks=len(relevant_chunks),
                answer=answer,
                processing_time=processing_time,
                metadata={
                    'movie_id': movie_id,
                    'distilled_context_length': len(distilled_context.distilled_text)
                }
            )
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"Pipeline completed in {processing_time:.2f}s")
            self.logger.info(f"{'='*80}\n")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            
            # Return error result
            processing_time = time.time() - start_time
            
            return RAGResult(
                question=question,
                route=QuestionRoute(category="UNKNOWN", requires_full_narrative=False),
                retrieved_chunks=0,
                relevant_chunks=0,
                answer=Answer(
                    question=question,
                    answer=f"An error occurred while processing your question: {str(e)}",
                    supporting_timestamps=[],
                    confidence="low",
                    source_chunk_ids=[],
                    model_used="error"
                ),
                processing_time=processing_time,
                metadata={'error': str(e)}
            )
    
    def _create_no_context_result(
        self,
        question: str,
        route: QuestionRoute,
        start_time: float
    ) -> RAGResult:
        """Create result when no context is found."""
        processing_time = time.time() - start_time
        
        return RAGResult(
            question=question,
            route=route,
            retrieved_chunks=0,
            relevant_chunks=0,
            answer=Answer(
                question=question,
                answer="I cannot find relevant information in the transcript to answer this question.",
                supporting_timestamps=[],
                confidence="low",
                source_chunk_ids=[],
                model_used="no_context"
            ),
            processing_time=processing_time,
            metadata={'error': 'no_context_found'}
        )
    
    def process_batch(
        self,
        questions: list,
        movie_id: Optional[str] = None
    ) -> list:
        """
        Process multiple questions.
        
        Args:
            questions: List of questions
            movie_id: Optional movie identifier
            
        Returns:
            List of RAGResult objects
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            self.logger.info(f"\nProcessing question {i}/{len(questions)}")
            result = self.process(question, movie_id)
            results.append(result)
        
        return results