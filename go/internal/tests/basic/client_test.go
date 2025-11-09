package basic

import (
	"testing"

	"github.com/drobyshevv/classifier-ai-agent/internal/client"
	"github.com/drobyshevv/classifier-ai-agent/internal/models"
)

// TestClientCreation проверяет создание клиента
func TestClientCreation(t *testing.T) {
	client := client.NewPythonMLClient("http://localhost:8000")
	if client == nil {
		t.Error("Client should not be nil")
	}
}

// TestRequestModel проверяет структуры данных
func TestRequestModel(t *testing.T) {
	req := &models.ArticleAnalysisRequest{
		DocumentID: "doc123",
		TitleRU:    "Заголовок",
		AbstractRU: "Описание",
	}

	if req.DocumentID != "doc123" {
		t.Error("DocumentID should be doc123")
	}
}
