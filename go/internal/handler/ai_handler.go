package handler

import (
	"context"

	"github.com/drobyshevv/classifier-ai-agent/internal/service"
	agentv1 "github.com/drobyshevv/proto-ai-agent/gen/go/proto/ai_agent"
)

type AIAnalysisHandler struct {
	agentv1.UnimplementedAIAnalysisServiceServer
	aiService *service.AIService
}

func NewAIAnalysisHandler(aiService *service.AIService) *AIAnalysisHandler {
	return &AIAnalysisHandler{
		aiService: aiService,
	}
}

func (h *AIAnalysisHandler) AnalyzeArticleTopics(ctx context.Context, req *agentv1.ArticleAnalysisRequest) (*agentv1.ArticleAnalysisResponse, error) {
	return h.aiService.AnalyzeArticleTopics(ctx, req)
}

func (h *AIAnalysisHandler) AnalyzeUserQuery(ctx context.Context, req *agentv1.QueryAnalysisRequest) (*agentv1.QueryAnalysisResponse, error) {
	return h.aiService.AnalyzeUserQuery(ctx, req)
}

func (h *AIAnalysisHandler) SemanticArticleSearch(ctx context.Context, req *agentv1.SemanticSearchRequest) (*agentv1.SemanticSearchResponse, error) {
	return h.aiService.SemanticArticleSearch(ctx, req)
}

func (h *AIAnalysisHandler) AnalyzeExpertsByTopic(ctx context.Context, req *agentv1.ExpertAnalysisRequest) (*agentv1.ExpertAnalysisResponse, error) {
	return h.aiService.AnalyzeExpertsByTopic(ctx, req)
}

func (h *AIAnalysisHandler) AnalyzeDepartmentsByTopic(ctx context.Context, req *agentv1.DepartmentAnalysisRequest) (*agentv1.DepartmentAnalysisResponse, error) {
	return h.aiService.AnalyzeDepartmentsByTopic(ctx, req)
}
