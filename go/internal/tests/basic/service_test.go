package basic

import (
	"testing"

	"github.com/drobyshevv/classifier-ai-agent/internal/client"
	"github.com/drobyshevv/classifier-ai-agent/internal/service"
)

// TestServiceCreation проверяет создание сервиса
func TestServiceCreation(t *testing.T) {
	pythonClient := client.NewPythonMLClient("http://localhost:8000")
	aiService := service.NewAIService(pythonClient)

	if aiService == nil {
		t.Error("Service should not be nil")
	}
}
