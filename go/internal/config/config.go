package config

type Config struct {
	Server struct {
		Host string `yaml:"host"`
		Port int    `yaml:"port"`
	} `yaml:"server"`

	PythonML struct {
		URL string `yaml:"url"` // "http://localhost:8000"
	} `yaml:"python_ml"`
}

func Load() *Config {
	cfg := &Config{}
	cfg.Server.Host = "0.0.0.0"
	cfg.Server.Port = 50051
	cfg.PythonML.URL = "http://localhost:8000"
	return cfg
}
