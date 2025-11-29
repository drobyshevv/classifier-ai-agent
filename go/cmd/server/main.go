package main

import (
	"fmt"
	"log"
	"net"

	"github.com/drobyshevv/classifier-ai-agent/config"
	"github.com/drobyshevv/classifier-ai-agent/internal/client"
	"github.com/drobyshevv/classifier-ai-agent/internal/handler"
	"github.com/drobyshevv/classifier-ai-agent/internal/service"
	agentv1 "github.com/drobyshevv/proto-ai-agent/gen/go/proto/ai_agent"
	"google.golang.org/grpc"
)

func main() {
	cfg := config.Load()

	// Клиент для Python ML сервиса
	pythonClient := client.NewPythonMLClient(cfg.PythonML.URL)

	// Сервисный слой
	aiService := service.NewAIService(pythonClient)

	// gRPC handler
	aiHandler := handler.NewAIAnalysisHandler(aiService)

	// Запуск gRPC сервера
	server := grpc.NewServer()
	agentv1.RegisterAIAnalysisServiceServer(server, aiHandler)

	address := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	lis, err := net.Listen("tcp", address)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	log.Printf("Go AI Agent server starting on %s", address)
	log.Printf("Python ML service: %s", cfg.PythonML.URL)

	if err := server.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
