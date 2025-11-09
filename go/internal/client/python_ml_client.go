package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/drobyshevv/classifier-ai-agent/internal/models"
)

// Клиент работы с Python
type PythonMLClient struct {
	baseURL    string
	httpClient *http.Client
}

func NewPythonMLClient(baseURL string) *PythonMLClient {
	return &PythonMLClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: time.Second * 60,
		},
	}
}

// AnalyzeArticleTopics - вызов Python для анализа статьи
func (c *PythonMLClient) AnalyzeArticleTopics(req *models.ArticleAnalysisRequest) (*models.ArticleAnalysisResponse, error) {
	url := fmt.Sprintf("%s/api/analyze-article", c.baseURL)

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to call Python ML service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Python ML service error: %s", string(body))
	}

	var response models.ArticleAnalysisResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// AnalyzeUserQuery - анализ пользовательского запроса
func (c *PythonMLClient) AnalyzeUserQuery(req *models.QueryAnalysisRequest) (*models.QueryAnalysisResponse, error) {
	url := fmt.Sprintf("%s/api/analyze-query", c.baseURL)

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to call Python ML service: %w", err)
	}
	defer resp.Body.Close()

	var response models.QueryAnalysisResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// SemanticArticleSearch - семантический поиск
func (c *PythonMLClient) SemanticArticleSearch(req *models.SemanticSearchRequest) (*models.SemanticSearchResponse, error) {
	url := fmt.Sprintf("%s/api/semantic-search", c.baseURL)

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to call Python ML service: %w", err)
	}
	defer resp.Body.Close()

	var response models.SemanticSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// AnalyzeExpertsByTopic - анализ экспертов
func (c *PythonMLClient) AnalyzeExpertsByTopic(req *models.ExpertAnalysisRequest) (*models.ExpertAnalysisResponse, error) {
	url := fmt.Sprintf("%s/api/analyze-experts", c.baseURL)

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to call Python ML service: %w", err)
	}
	defer resp.Body.Close()

	var response models.ExpertAnalysisResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}

// AnalyzeDepartmentsByTopic - анализ кафедр
func (c *PythonMLClient) AnalyzeDepartmentsByTopic(req *models.DepartmentAnalysisRequest) (*models.DepartmentAnalysisResponse, error) {
	url := fmt.Sprintf("%s/api/analyze-departments", c.baseURL)

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(url, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to call Python ML service: %w", err)
	}
	defer resp.Body.Close()

	var response models.DepartmentAnalysisResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &response, nil
}
