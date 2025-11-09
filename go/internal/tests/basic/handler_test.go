package basic

import (
	"testing"

	"github.com/drobyshevv/classifier-ai-agent/internal/client"
	"github.com/drobyshevv/classifier-ai-agent/internal/handler"
	"github.com/drobyshevv/classifier-ai-agent/internal/service"
)

// TestHandlerCreation проверяет создание handler
func TestHandlerCreation(t *testing.T) {
	pythonClient := client.NewPythonMLClient("http://localhost:8000")
	aiService := service.NewAIService(pythonClient)
	handler := handler.NewAIAnalysisHandler(aiService)

	if handler == nil {
		t.Error("Handler should not be nil")
	}
}
