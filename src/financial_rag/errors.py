"""Domain-specific errors with actionable failure context."""


class FinancialRAGError(Exception):
    """Base class for project errors."""


class SECDownloadError(FinancialRAGError):
    """Raised when SEC metadata or filing content cannot be downloaded."""


class FilingParseError(FinancialRAGError):
    """Raised when a downloaded filing cannot be parsed into usable text."""


class EmbeddingError(FinancialRAGError):
    """Raised when embedding generation returns unusable vectors."""


class IndexBuildError(FinancialRAGError):
    """Raised when a dense index cannot be built or loaded safely."""


class RetrievalError(FinancialRAGError):
    """Raised when a dense retrieval query cannot be completed."""


class RerankingError(FinancialRAGError):
    """Raised when cross-encoder reranking cannot be completed."""


class FinancialIntelligenceError(FinancialRAGError):
    """Raised when structured financial evidence cannot be loaded or interpreted."""


class ReproductionError(FinancialRAGError):
    """Raised when frozen benchmark inputs cannot be reproduced exactly."""
