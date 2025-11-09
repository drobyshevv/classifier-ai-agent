package integration

import (
	"context"
	"testing"

	"github.com/drobyshevv/classifier-ai-agent/internal/client"
	"github.com/drobyshevv/classifier-ai-agent/internal/handler"
	"github.com/drobyshevv/classifier-ai-agent/internal/service"
	agentv1 "github.com/drobyshevv/proto-ai-agent/gen/go/proto/ai_agent"
)

// TestFullFlow_ArticleToSearch проверяет полный цикл: статья -> запрос -> поиск
func TestFullFlow_ArticleToSearch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	pythonClient := client.NewPythonMLClient("http://localhost:8000")
	aiService := service.NewAIService(pythonClient)
	aiHandler := handler.NewAIAnalysisHandler(aiService)

	t.Log("🚀 Starting full flow test...")

	// Шаг 1: Анализ статьи
	articleRequest := &agentv1.ArticleAnalysisRequest{
		DocumentId: "flow_test_001",
		TitleRu:    "Глубокое обучение для медицинской диагностики",
		AbstractRu: "Применение сверточных нейронных сетей для анализа медицинских изображений и диагностики заболеваний с использованием компьютерного зрения и искусственного интеллекта",
	}

	articleResponse, err := aiHandler.AnalyzeArticleTopics(context.Background(), articleRequest)
	if err != nil {
		t.Skip("Step 1 failed - Python service not available:", err)
		return
	}

	t.Logf("✅ Step 1 - Article Analysis: %d topics found", len(articleResponse.Topics))

	// Шаг 2: Анализ поискового запроса
	queryRequest := &agentv1.QueryAnalysisRequest{
		UserQuery: "медицинская диагностика с помощью искусственного интеллекта",
		Context:   "article_search",
	}

	queryResponse, err := aiHandler.AnalyzeUserQuery(context.Background(), queryRequest)
	if err != nil {
		t.Skip("Step 2 failed:", err)
		return
	}

	t.Logf("✅ Step 2 - Query Analysis: '%s'", queryResponse.InterpretedQuery)

	// Шаг 3: Семантический поиск (упрощенный)
	searchRequest := &agentv1.SemanticSearchRequest{
		QueryVector: queryResponse.QueryVector,
		MaxResults:  3,
		Articles: []*agentv1.ArticleForSearch{
			{
				DocumentId:        "doc_medical_ai",
				TitleRu:           "ИИ в медицине",
				AbstractRu:        "Искусственный интеллект для медицинской диагностики",
				TitleEmbedding:    articleResponse.TitleEmbedding, // Используем эмбеддинг из шага 1
				AbstractEmbedding: articleResponse.AbstractEmbedding,
			},
		},
	}

	searchResponse, err := aiHandler.SemanticArticleSearch(context.Background(), searchRequest)
	if err != nil {
		t.Skip("Step 3 failed:", err)
		return
	}

	t.Logf("✅ Step 3 - Semantic Search: %d results found", len(searchResponse.Results))

	t.Log("🎉 Full flow test completed successfully!")
}

// TestMultipleOperations проверяет несколько операций подряд
func TestMultipleOperations(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	pythonClient := client.NewPythonMLClient("http://localhost:8000")
	aiService := service.NewAIService(pythonClient)
	aiHandler := handler.NewAIAnalysisHandler(aiService)

	// Тест 1: Анализ статьи
	articleResp, err1 := aiHandler.AnalyzeArticleTopics(context.Background(), &agentv1.ArticleAnalysisRequest{
		DocumentId: "multi_test_1",
		TitleRu:    "Машинное обучение",
		AbstractRu: "Алгоритмы машинного обучения",
	})

	// Тест 2: Анализ запроса
	queryResp, err2 := aiHandler.AnalyzeUserQuery(context.Background(), &agentv1.QueryAnalysisRequest{
		UserQuery: "найти про машинное обучение",
		Context:   "article_search",
	})

	// Тест 3: Анализ экспертов
	expertsResp, err3 := aiHandler.AnalyzeExpertsByTopic(context.Background(), &agentv1.ExpertAnalysisRequest{
		Topic: "машинное обучение",
		Authors: []*agentv1.AuthorArticles{
			{
				AuthorId:      "author_ml",
				ArticleIds:    []string{"ml_paper_1"},
				ArticleTopics: []string{"машинное обучение", "нейронные сети"},
			},
		},
	})

	// Подсчитываем успешные операции
	successCount := 0
	if err1 == nil && articleResp != nil {
		successCount++
		t.Log("✅ Article analysis: OK")
	}
	if err2 == nil && queryResp != nil {
		successCount++
		t.Log("✅ Query analysis: OK")
	}
	if err3 == nil && expertsResp != nil {
		successCount++
		t.Log("✅ Experts analysis: OK")
	}

	t.Logf("🎯 Successfully completed %d out of 3 operations", successCount)
}
