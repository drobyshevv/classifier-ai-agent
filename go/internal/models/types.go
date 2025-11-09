package models

// Модели для HTTP взаимодействия с Python ML сервисом

type ArticleAnalysisRequest struct {
	DocumentID string `json:"document_id"`
	TitleRU    string `json:"title_ru"`
	AbstractRU string `json:"abstract_ru"`
}

type ArticleTopic struct {
	TopicName  string  `json:"topic_name"`
	Confidence float32 `json:"confidence"`
	TopicType  string  `json:"topic_type"`
}

type ArticleAnalysisResponse struct {
	Topics            []ArticleTopic `json:"topics"`
	TitleEmbedding    []byte         `json:"title_embedding"`
	AbstractEmbedding []byte         `json:"abstract_embedding"`
}

type QueryAnalysisRequest struct {
	UserQuery string `json:"user_query"`
	Context   string `json:"context"`
}

type QueryAnalysisResponse struct {
	InterpretedQuery string   `json:"interpreted_query"`
	KeyConcepts      []string `json:"key_concepts"`
	QueryVector      []byte   `json:"query_vector"`
	QueryType        string   `json:"query_type"`
}

type ArticleForSearch struct {
	DocumentID        string `json:"document_id"`
	TitleRU           string `json:"title_ru"`
	AbstractRU        string `json:"abstract_ru"`
	TitleEmbedding    []byte `json:"title_embedding"`
	AbstractEmbedding []byte `json:"abstract_embedding"`
}

type SemanticSearchRequest struct {
	QueryVector []byte             `json:"query_vector"`
	Articles    []ArticleForSearch `json:"articles"`
	MaxResults  int32              `json:"max_results"`
}

type SearchResult struct {
	DocumentID      string   `json:"document_id"`
	RelevanceScore  float32  `json:"relevance_score"`
	MatchedConcepts []string `json:"matched_concepts"`
}

type SemanticSearchResponse struct {
	Results    []SearchResult `json:"results"`
	TotalFound int32          `json:"total_found"`
}

type AuthorArticles struct {
	AuthorID      string   `json:"author_id"`
	ArticleIDs    []string `json:"article_ids"`
	ArticleTopics []string `json:"article_topics"`
}

type ExpertAnalysisRequest struct {
	Topic   string           `json:"topic"`
	Authors []AuthorArticles `json:"authors"`
}

type ExpertAnalysis struct {
	AuthorID          string   `json:"author_id"`
	ExpertiseScore    float32  `json:"expertise_score"`
	TopicArticleCount int32    `json:"topic_article_count"`
	TotalCitations    int32    `json:"total_citations"`
	LastActivityYear  int32    `json:"last_activity_year"`
	RelatedTopics     []string `json:"related_topics"`
}

type ExpertAnalysisResponse struct {
	Experts []ExpertAnalysis `json:"experts"`
}

type DepartmentData struct {
	OrganizationID string   `json:"organization_id"`
	AuthorIDs      []string `json:"author_ids"`
	ArticleTopics  []string `json:"article_topics"`
}

type DepartmentAnalysisRequest struct {
	Topic       string           `json:"topic"`
	Departments []DepartmentData `json:"departments"`
}

type DepartmentAnalysis struct {
	OrganizationID string   `json:"organization_id"`
	StrengthScore  float32  `json:"strength_score"`
	ExpertCount    int32    `json:"expert_count"`
	TotalArticles  int32    `json:"total_articles"`
	KeyAuthorIDs   []string `json:"key_author_ids"`
}

type DepartmentAnalysisResponse struct {
	Departments []DepartmentAnalysis `json:"departments"`
}
