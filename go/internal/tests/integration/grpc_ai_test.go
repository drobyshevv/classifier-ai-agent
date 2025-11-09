package integration

import (
	"context"
	"testing"

	"github.com/drobyshevv/classifier-ai-agent/internal/client"
	"github.com/drobyshevv/classifier-ai-agent/internal/handler"
	"github.com/drobyshevv/classifier-ai-agent/internal/service"
	agentv1 "github.com/drobyshevv/proto-ai-agent/gen/go/proto/ai_agent"
)

// TestGRPCArticleAnalysis проверяет полный цикл: gRPC -> Go -> Python
func TestGRPCArticleAnalysis(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Создаем всю цепочку
	pythonClient := client.NewPythonMLClient("http://localhost:8000")
	aiService := service.NewAIService(pythonClient)
	aiHandler := handler.NewAIAnalysisHandler(aiService)

	// Тестовый запрос
	request := &agentv1.ArticleAnalysisRequest{
		DocumentId: "integration_grpc_001",
		TitleRu:    "gRPC интеграционное тестирование с Python ML",
		AbstractRu: "Проверка полного цикла обработки запроса от gRPC клиента через Go сервис к Python ML и обратно",
	}

	// Вызываем handler напрямую (без запуска gRPC сервера)
	response, err := aiHandler.AnalyzeArticleTopics(context.Background(), request)

	// Проверяем результат
	if err != nil {
		t.Skip("Python ML service not available:", err)
		return
	}

	// Базовые проверки ответа
	if response == nil {
		t.Error("Response should not be nil")
		return
	}

	if len(response.Topics) == 0 {
		t.Error("Should return at least one topic")
	}

	if len(response.TitleEmbedding) == 0 {
		t.Error("Should return title embedding")
	}

	t.Logf("✅ gRPC Article Analysis: %d topics, embedding size: %d bytes",
		len(response.Topics), len(response.TitleEmbedding))
}

// TestGRPCQueryAnalysis проверяет анализ пользовательского запроса
func TestGRPCQueryAnalysis(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	pythonClient := client.NewPythonMLClient("http://localhost:8000")
	aiService := service.NewAIService(pythonClient)
	aiHandler := handler.NewAIAnalysisHandler(aiService)

	request := &agentv1.QueryAnalysisRequest{
		UserQuery: "найти статьи про искусственный интеллект и машинное обучение",
		Context:   "article_search",
	}

	response, err := aiHandler.AnalyzeUserQuery(context.Background(), request)

	if err != nil {
		t.Skip("Python ML service not available:", err)
		return
	}

	if response == nil {
		t.Error("Response should not be nil")
		return
	}

	if response.InterpretedQuery == "" {
		t.Error("Should return interpreted query")
	}

	t.Logf("✅ gRPC Query Analysis: '%s' -> '%s'",
		request.UserQuery, response.InterpretedQuery)
}

// TestGRPCExpertsAnalysis проверяет анализ экспертов
func TestGRPCExpertsAnalysis(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	pythonClient := client.NewPythonMLClient("http://localhost:8000")
	aiService := service.NewAIService(pythonClient)
	aiHandler := handler.NewAIAnalysisHandler(aiService)

	request := &agentv1.ExpertAnalysisRequest{
		Topic: "компьютерное зрение",
		Authors: []*agentv1.AuthorArticles{
			{
				AuthorId:      "expert_001",
				ArticleIds:    []string{"cv_paper_1", "cv_paper_2"},
				ArticleTopics: []string{"компьютерное зрение", "нейронные сети", "обработка изображений"},
			},
		},
	}

	response, err := aiHandler.AnalyzeExpertsByTopic(context.Background(), request)

	if err != nil {
		t.Skip("Python ML service not available:", err)
		return
	}

	if response == nil {
		t.Error("Response should not be nil")
		return
	}

	t.Logf("✅ gRPC Experts Analysis: %d experts found", len(response.Experts))
}
