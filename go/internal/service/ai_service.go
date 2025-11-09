package service

import (
	"context"
	"fmt"

	"github.com/drobyshevv/classifier-ai-agent/internal/client"
	"github.com/drobyshevv/classifier-ai-agent/internal/models"
	agentv1 "github.com/drobyshevv/proto-ai-agent/gen/go/proto/ai_agent"
)

type AIService struct {
	pythonClient *client.PythonMLClient
}

func NewAIService(pythonClient *client.PythonMLClient) *AIService {
	return &AIService{
		pythonClient: pythonClient,
	}
}

func (s *AIService) AnalyzeArticleTopics(ctx context.Context, req *agentv1.ArticleAnalysisRequest) (*agentv1.ArticleAnalysisResponse, error) {
	httpReq := &models.ArticleAnalysisRequest{
		DocumentID: req.DocumentId,
		TitleRU:    req.TitleRu,
		AbstractRU: req.AbstractRu,
	}

	httpResp, err := s.pythonClient.AnalyzeArticleTopics(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Python ML service error: %w", err)
	}

	resp := &agentv1.ArticleAnalysisResponse{
		TitleEmbedding:    httpResp.TitleEmbedding,
		AbstractEmbedding: httpResp.AbstractEmbedding,
	}

	for _, topic := range httpResp.Topics {
		resp.Topics = append(resp.Topics, &agentv1.ArticleTopic{
			TopicName:  topic.TopicName,
			Confidence: topic.Confidence,
			TopicType:  topic.TopicType,
		})
	}

	return resp, nil
}

func (s *AIService) AnalyzeUserQuery(ctx context.Context, req *agentv1.QueryAnalysisRequest) (*agentv1.QueryAnalysisResponse, error) {
	httpReq := &models.QueryAnalysisRequest{
		UserQuery: req.UserQuery,
		Context:   req.Context,
	}

	httpResp, err := s.pythonClient.AnalyzeUserQuery(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Python ML service error: %w", err)
	}

	resp := &agentv1.QueryAnalysisResponse{
		InterpretedQuery: httpResp.InterpretedQuery,
		QueryVector:      httpResp.QueryVector,
		QueryType:        httpResp.QueryType,
	}

	for _, concept := range httpResp.KeyConcepts {
		resp.KeyConcepts = append(resp.KeyConcepts, concept)
	}

	return resp, nil
}

func (s *AIService) SemanticArticleSearch(ctx context.Context, req *agentv1.SemanticSearchRequest) (*agentv1.SemanticSearchResponse, error) {
	httpReq := &models.SemanticSearchRequest{
		QueryVector: req.QueryVector,
		MaxResults:  req.MaxResults,
	}

	for _, article := range req.Articles {
		httpReq.Articles = append(httpReq.Articles, models.ArticleForSearch{
			DocumentID:        article.DocumentId,
			TitleRU:           article.TitleRu,
			AbstractRU:        article.AbstractRu,
			TitleEmbedding:    article.TitleEmbedding,
			AbstractEmbedding: article.AbstractEmbedding,
		})
	}

	httpResp, err := s.pythonClient.SemanticArticleSearch(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Python ML service error: %w", err)
	}

	resp := &agentv1.SemanticSearchResponse{
		TotalFound: httpResp.TotalFound,
	}

	for _, result := range httpResp.Results {
		resultPB := &agentv1.SearchResult{
			DocumentId:     result.DocumentID,
			RelevanceScore: result.RelevanceScore,
		}

		for _, concept := range result.MatchedConcepts {
			resultPB.MatchedConcepts = append(resultPB.MatchedConcepts, concept)
		}

		resp.Results = append(resp.Results, resultPB)
	}

	return resp, nil
}

func (s *AIService) AnalyzeExpertsByTopic(ctx context.Context, req *agentv1.ExpertAnalysisRequest) (*agentv1.ExpertAnalysisResponse, error) {
	httpReq := &models.ExpertAnalysisRequest{
		Topic: req.Topic,
	}

	for _, author := range req.Authors {
		httpReq.Authors = append(httpReq.Authors, models.AuthorArticles{
			AuthorID:      author.AuthorId,
			ArticleIDs:    author.ArticleIds,
			ArticleTopics: author.ArticleTopics,
		})
	}

	httpResp, err := s.pythonClient.AnalyzeExpertsByTopic(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Python ML service error: %w", err)
	}

	resp := &agentv1.ExpertAnalysisResponse{}

	for _, expert := range httpResp.Experts {
		expertPB := &agentv1.ExpertAnalysis{
			AuthorId:          expert.AuthorID,
			ExpertiseScore:    expert.ExpertiseScore,
			TopicArticleCount: expert.TopicArticleCount,
			TotalCitations:    expert.TotalCitations,
			LastActivityYear:  expert.LastActivityYear,
		}

		for _, topic := range expert.RelatedTopics {
			expertPB.RelatedTopics = append(expertPB.RelatedTopics, topic)
		}

		resp.Experts = append(resp.Experts, expertPB)
	}

	return resp, nil
}

func (s *AIService) AnalyzeDepartmentsByTopic(ctx context.Context, req *agentv1.DepartmentAnalysisRequest) (*agentv1.DepartmentAnalysisResponse, error) {
	httpReq := &models.DepartmentAnalysisRequest{
		Topic: req.Topic,
	}

	for _, dept := range req.Departments {
		httpReq.Departments = append(httpReq.Departments, models.DepartmentData{
			OrganizationID: dept.OrganizationId,
			AuthorIDs:      dept.AuthorIds,
			ArticleTopics:  dept.ArticleTopics,
		})
	}

	httpResp, err := s.pythonClient.AnalyzeDepartmentsByTopic(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Python ML service error: %w", err)
	}

	resp := &agentv1.DepartmentAnalysisResponse{}

	for _, dept := range httpResp.Departments {
		deptPB := &agentv1.DepartmentAnalysis{
			OrganizationId: dept.OrganizationID,
			StrengthScore:  dept.StrengthScore,
			ExpertCount:    dept.ExpertCount,
			TotalArticles:  dept.TotalArticles,
		}

		for _, authorID := range dept.KeyAuthorIDs {
			deptPB.KeyAuthorIds = append(deptPB.KeyAuthorIds, authorID)
		}

		resp.Departments = append(resp.Departments, deptPB)
	}

	return resp, nil
}
