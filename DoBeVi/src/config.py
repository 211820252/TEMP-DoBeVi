import os
from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Union, List

class Settings(BaseSettings):
    REPO_PATH: str = Field('/absolute/path/to/lean/repo', description="Absolute path to the Lean project repo")
    FILE_PATHS: Union[str, List[str]] = Field(
        default='relative/path/to/lean/file',
        description="Single or multiple relative paths to the Lean theorem file"
    )
    MODEL_PATH: str = Field('path/to/model', description="Path to the LLM or HuggingFace model")
    ALGORITHM: str = Field('best_first', description="Search algorithm to use")
    NUM_WORKERS: int = Field(2, description="Number of concurrent worker processes")
    NUM_GPUS: int = Field(1, description="Number of GPUs to use")
    NUM_SAMPLED_TACTICS: int = Field(4, description="Number of tactics to sample at each step")
    SEARCH_TIMEOUT: int = Field(1800, description="Time limit for the proof search process")
    MAX_EXPANSIONS: int | None = Field(None, description="Maximum number of expansions during proof search")
    RESULT_SAVE_PATH: str = Field("../results", description="Path to save results")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DOBEVI_"
    )

    @field_validator("FILE_PATHS", mode="before")
    @classmethod
    def validate_file_paths(cls, v: str) -> Union[str, List[str]]:
        if ',' in v:
            return [p.strip() for p in v.split(',')]
        return v

    @field_validator("ALGORITHM", mode="before")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        allowed = ['best_first', 'group_score', 'internlm_bfs', 'layer_dropout']
        if v in allowed:
            return v
        if v == "":
            return 'best_first'
        raise ValidationError(f"Invalid ALGORITHM: {v!r}. Must be one of {allowed}.")
    
    @field_validator("NUM_WORKERS", mode="before")
    @classmethod
    def validate_num_workers(cls, v: str) -> str:
        return 2 if v == "" else v 

    @field_validator("NUM_GPUS", mode="before")
    @classmethod
    def validate_num_gpus(cls, v: str) -> str:
        return 1 if v == "" else v 

    @field_validator("NUM_SAMPLED_TACTICS", mode="before")
    @classmethod
    def validate_num_sampled_tactics(cls, v: str) -> str:
        return 4 if v == "" else v 

    @field_validator("SEARCH_TIMEOUT", mode="before")
    @classmethod
    def validate_search_timeout(cls, v: str) -> str:
        return 1800 if v == "" else v 

    @field_validator("MAX_EXPANSIONS", mode="before")
    @classmethod
    def validate_max_expansions(cls, v: str) -> str:
        return None if v == "" else v

    @field_validator("RESULT_SAVE_PATH", mode="before")
    @classmethod
    def validate_result_save_path(cls, v: str) -> str:
        return "../results" if v == "" else v


# Load settings with error reporting
try:
    settings = Settings()
except ValidationError as e:
    raise ValueError(f"Environment variable validation failed: {e}")
