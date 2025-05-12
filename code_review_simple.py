#!/usr/bin/env python3
"""
Code Review Framework using Crew AI Agentic Approach
"""

import os
import sys
import yaml
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import git
import requests
from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import tempfile
import shutil

# Create necessary directories
os.makedirs("agents", exist_ok=True)
os.makedirs("tasks", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Agent configurations
AGENTS_CONFIG = {
    "agents": {
        "code_standards_reviewer": {
            "role": "Code Standards Reviewer",
            "goal": "Review code for adherence to language-specific standards and best practices",
            "backstory": "You are an experienced software engineer with expertise in multiple programming languages and their respective coding standards.",
            "tools": ["code_analysis"],
            "verbose": True
        },
        "performance_analyzer": {
            "role": "Performance Analyzer",
            "goal": "Identify performance bottlenecks and optimization opportunities",
            "backstory": "You specialize in code performance analysis and optimization across various programming languages.",
            "tools": ["performance_analysis"],
            "verbose": True
        },
        "security_auditor": {
            "role": "Security Auditor",
            "goal": "Identify security vulnerabilities and compliance issues",
            "backstory": "You are a cybersecurity expert with deep knowledge of OWASP guidelines and secure coding practices.",
            "tools": ["security_analysis"],
            "verbose": True
        },
        "documentation_reviewer": {
            "role": "Documentation Reviewer",
            "goal": "Review code documentation and suggest improvements",
            "backstory": "You specialize in technical documentation and ensuring code readability through proper comments and documentation.",
            "tools": ["documentation_analysis"],
            "verbose": True
        },
        "reporting_agent": {
            "role": "Report Generator",
            "goal": "Aggregate findings and generate comprehensive reports",
            "backstory": "You are responsible for synthesizing multiple code review results into clear, actionable reports.",
            "tools": ["report_generation"],
            "verbose": True
        },
        "supervisor": {
            "role": "Code Review Supervisor",
            "goal": "Orchestrate the code review process and ensure all aspects are covered",
            "backstory": "You coordinate multiple specialized agents to perform a comprehensive code review.",
            "tools": ["task_management"],
            "verbose": True
        }
    }
}

# Task configurations
TASKS_CONFIG = {
    "tasks": {
        "analyze_standards": {
            "description": "Analyze code for adherence to language-specific standards",
            "expected_output": "Detailed report on coding standards violations and recommendations",
            "agent": "code_standards_reviewer"
        },
        "analyze_performance": {
            "description": "Analyze code for performance issues and optimization opportunities",
            "expected_output": "Performance analysis report with specific recommendations",
            "agent": "performance_analyzer"
        },
        "audit_security": {
            "description": "Perform security audit on the codebase",
            "expected_output": "Security audit report with identified vulnerabilities and fixes",
            "agent": "security_auditor"
        },
        "review_documentation": {
            "description": "Review code documentation and suggest improvements",
            "expected_output": "Documentation review report with suggestions",
            "agent": "documentation_reviewer"
        },
        "generate_report": {
            "description": "Generate comprehensive code review report",
            "expected_output": "Final code review report in PDF format",
            "agent": "reporting_agent"
        }
    }
}

# Save configurations to YAML files
with open("agents/agents.yaml", "w") as f:
    yaml.dump(AGENTS_CONFIG, f, default_flow_style=False)

with open("tasks/tasks.yaml", "w") as f:
    yaml.dump(TASKS_CONFIG, f, default_flow_style=False)

class LLMClient:
    """Local LLM endpoint client for Qwen2.5-Coder"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config["apibase"]
        self.model = config["model"]
        self.headers = {
            "Content-Type": "application/json"
        }
        if config.get("apikey"):
            self.headers["Authorization"] = f"Bearer {config['apikey']}"
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate response using the local LLM endpoint"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            **self.config.get("parameters", {})
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
        
        except Exception as e:
            return f"Error: {str(e)}"

class CodeChunker:
    """Handles code chunking for large files"""
    
    def __init__(self, chunk_size: int = 2000):
        self.chunk_size = chunk_size
    
    def chunk_code(self, content: str, language: str = "python") -> List[Dict[str, Any]]:
        """Chunk code into manageable pieces"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            current_size += len(line) + 1  # +1 for newline
            
            if current_size >= self.chunk_size or i == len(lines) - 1:
                chunk_content = '\n'.join(current_chunk)
                chunks.append({
                    "chunk_id": len(chunks) + 1,
                    "content": chunk_content,
                    "start_line": i + 1 - len(current_chunk) + 1,
                    "end_line": i + 1,
                    "language": language
                })
                current_chunk = []
                current_size = 0
        
        return chunks

class CodeReviewAgent:
    """Base agent class for code review"""
    
    def __init__(self, role: str, goal: str, backstory: str, llm_client: LLMClient):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.llm_client = llm_client
    
    def review_chunk(self, chunk: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Review a single chunk of code"""
        prompt_template = PromptTemplate(
            input_variables=["role", "goal", "backstory", "language", "file_path", "chunk_content"],
            template="""You are a {role}.
Goal: {goal}
Backstory: {backstory}

Analyze the following {language} code from {file_path}:

```{language}
{chunk_content}
```

Provide a detailed review focusing on:
1. Coding standards and best practices
2. Performance optimization opportunities
3. Security vulnerabilities
4. Documentation quality

Format your response as JSON with the following structure:
{{
    "findings": [
        {{
            "type": "standards|performance|security|documentation",
            "severity": "critical|high|medium|low",
            "description": "Description of the issue",
            "line_number": 0,
            "recommendation": "Specific recommendation"
        }}
    ],
    "summary": "Overall summary of findings"
}}"""
        )
        
        prompt = prompt_template.format(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            language=chunk["language"],
            file_path=file_path,
            chunk_content=chunk["content"]
        )
        
        system_prompt = f"You are a {self.role} specializing in code review. Always respond in valid JSON format."
        response = self.llm_client.generate(prompt, system_prompt)
        
        try:
            # Parse JSON response
            return json.loads(response)
        except:
            # Fallback if JSON parsing fails
            return {
                "findings": [],
                "summary": response,
                "error": "Failed to parse JSON response"
            }

class SupervisorAgent:
    """Orchestrates the code review process"""
    
    def __init__(self, agents: Dict[str, CodeReviewAgent], llm_client: LLMClient):
        self.agents = agents
        self.llm_client = llm_client
        self.chunker = CodeChunker()
    
    def review_repository(self, repo_path: str) -> Dict[str, Any]:
        """Perform comprehensive review of the repository"""
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "repository": repo_path,
                "agents": list(self.agents.keys())
            },
            "files": {},
            "summary": {}
        }
        
        # Analyze all files in the repository
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                # Determine file type
                _, ext = os.path.splitext(file)
                language = self._detect_language(ext)
                
                if language and self._is_source_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Chunk the file
                        chunks = self.chunker.chunk_code(content, language)
                        
                        # Review each chunk with all agents
                        file_results = {
                            "language": language,
                            "chunks": len(chunks),
                            "reviews": {}
                        }
                        
                        for agent_name, agent in self.agents.items():
                            if agent_name == "supervisor":
                                continue
                            
                            chunk_reviews = []
                            for chunk in chunks:
                                review = agent.review_chunk(chunk, relative_path)
                                chunk_reviews.append(review)
                            
                            file_results["reviews"][agent_name] = chunk_reviews
                        
                        results["files"][relative_path] = file_results
                        
                    except Exception as e:
                        print(f"Error reviewing {relative_path}: {str(e)}")
        
        # Generate overall summary
        results["summary"] = self._generate_summary(results)
        
        return results
    
    def _detect_language(self, extension: str) -> Optional[str]:
        """Detect programming language from file extension"""
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".sql": "sql",
            ".html": "html",
            ".css": "css",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".xml": "xml",
            ".sh": "bash",
            ".md": "markdown"
        }
        return language_map.get(extension.lower())
    
    def _is_source_file(self, file_path: str) -> bool:
        """Check if file is a source code file"""
        # Basic check - you can expand this
        return not any(
            file_path.endswith(ext) for ext in [
                '.pyc', '.class', '.o', '.so', '.dll', '.exe', 
                '.png', '.jpg', '.jpeg', '.gif', '.pdf', '.zip'
            ]
        )
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary of findings"""
        summary = {
            "total_files": len(results["files"]),
            "total_issues": 0,
            "issues_by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "issues_by_type": {"standards": 0, "performance": 0, "security": 0, "documentation": 0},
            "top_issues": []
        }
        
        all_issues = []
        
        for file_path, file_data in results["files"].items():
            for agent_name, reviews in file_data.get("reviews", {}).items():
                for review in reviews:
                    if "findings" in review:
                        for finding in review["findings"]:
                            issue = {
                                "file": file_path,
                                "agent": agent_name,
                                "type": finding.get("type", "unknown"),
                                "severity": finding.get("severity", "low"),
                                "description": finding.get("description", ""),
                                "recommendation": finding.get("recommendation", ""),
                                "line_number": finding.get("line_number", 0)
                            }
                            all_issues.append(issue)
                            summary["total_issues"] += 1
                            
                            # Count by severity
                            severity = finding.get("severity", "low")
                            if severity in summary["issues_by_severity"]:
                                summary["issues_by_severity"][severity] += 1
                            
                            # Count by type
                            issue_type = finding.get("type", "unknown")
                            if issue_type in summary["issues_by_type"]:
                                summary["issues_by_type"][issue_type] += 1
        
        # Get top 10 most critical issues
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        summary["top_issues"] = sorted(
            all_issues, 
            key=lambda x: (severity_order.get(x["severity"], 3), x["file"])
        )[:10]
        
        return summary

class ReportGenerator:
    """Generates PDF reports from code review results"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def generate_pdf_report(self, results: Dict[str, Any], filename: str = "code_review_report.pdf") -> str:
        """Generate a comprehensive PDF report"""
        output_path = os.path.join(self.output_dir, filename)
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph("Code Review Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Metadata
        metadata = results.get("metadata", {})
        para = Paragraph(f"<b>Repository:</b> {metadata.get('repository', 'Unknown')}", styles['Normal'])
        story.append(para)
        para = Paragraph(f"<b>Generated:</b> {metadata.get('timestamp', 'Unknown')}", styles['Normal'])
        story.append(para)
        story.append(Spacer(1, 12))
        
        # Summary section
        story.append(Paragraph("Executive Summary", styles['Heading1']))
        summary = results.get("summary", {})
        
        data = [
            ["Metric", "Value"],
            ["Total Files Analyzed", str(summary.get("total_files", 0))],
            ["Total Issues Found", str(summary.get("total_issues", 0))],
            ["Critical Issues", str(summary.get("issues_by_severity", {}).get("critical", 0))],
            ["High Priority Issues", str(summary.get("issues_by_severity", {}).get("high", 0))],
            ["Medium Priority Issues", str(summary.get("issues_by_severity", {}).get("medium", 0))],
            ["Low Priority Issues", str(summary.get("issues_by_severity", {}).get("low", 0))]
        ]
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Top Issues section
        story.append(Paragraph("Top Critical Issues", styles['Heading1']))
        
        for i, issue in enumerate(summary.get("top_issues", [])[:5], 1):
            story.append(Paragraph(f"Issue #{i}", styles['Heading2']))
            story.append(Paragraph(f"<b>File:</b> {issue.get('file', 'Unknown')}", styles['Normal']))
            story.append(Paragraph(f"<b>Severity:</b> {issue.get('severity', 'Unknown')}", styles['Normal']))
            story.append(Paragraph(f"<b>Type:</b> {issue.get('type', 'Unknown')}", styles['Normal']))
            story.append(Paragraph(f"<b>Description:</b> {issue.get('description', 'No description')}", styles['Normal']))
            story.append(Paragraph(f"<b>Recommendation:</b> {issue.get('recommendation', 'No recommendation')}", styles['Normal']))
            if issue.get('line_number', 0) > 0:
                story.append(Paragraph(f"<b>Line:</b> {issue.get('line_number', 0)}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Build the PDF
        doc.build(story)
        return output_path

def clone_github_repo(repo_url: str, access_token: Optional[str] = None, target_dir: Optional[str] = None) -> str:
    """Clone a GitHub repository"""
    if target_dir is None:
        target_dir = tempfile.mkdtemp()
    
    # Add token to URL if provided
    if access_token and "github.com" in repo_url:
        if "https://github.com" in repo_url:
            repo_url = repo_url.replace("https://github.com", f"https://{access_token}@github.com")
        elif "http://github.com" in repo_url:
            repo_url = repo_url.replace("http://github.com", f"http://{access_token}@github.com")
    
    try:
        print(f"Cloning repository to {target_dir}...")
        git.Repo.clone_from(repo_url, target_dir)
        print("Repository cloned successfully!")
        return target_dir
    except Exception as e:
        print(f"Error cloning repository: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Code Review Framework using Crew AI")
    parser.add_argument("--repo", required=True, help="GitHub repository URL")
    parser.add_argument("--token", help="GitHub access token")
    parser.add_argument("--output-dir", default="output", help="Output directory for reports")
    parser.add_argument("--llm-endpoint", default="http://localhost:8000", help="LLM endpoint URL")
    parser.add_argument("--model", default="qwen2.5-coder", help="LLM model name")
    
    args = parser.parse_args()
    
    # Configure LLM client
    llm_config = {
        "title": "Local Qwen2.5",
        "provider": "Qwen",
        "model": args.model,
        "apikey": "",
        "apibase": args.llm_endpoint,
        "parameters": {
            "max_tokens": 2048,
            "temperature": 0.7
        }
    }
    
    llm_client = LLMClient(llm_config)
    
    # Create agents
    agents = {}
    for agent_name, agent_config in AGENTS_CONFIG["agents"].items():
        if agent_name != "supervisor":
            agents[agent_name] = CodeReviewAgent(
                role=agent_config["role"],
                goal=agent_config["goal"],
                backstory=agent_config["backstory"],
                llm_client=llm_client
            )
    
    # Create supervisor
    supervisor = SupervisorAgent(agents, llm_client)
    
    # Clone repository
    repo_dir = None
    try:
        repo_dir = clone_github_repo(args.repo, args.token)
        
        # Perform code review
        print("Starting code review process...")
        results = supervisor.review_repository(repo_dir)
        
        # Generate report
        os.makedirs(args.output_dir, exist_ok=True)
        report_generator = ReportGenerator(args.output_dir)
        
        # Generate PDF report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_name = args.repo.split('/')[-1].replace('.git', '')
        pdf_filename = f"{repo_name}_code_review_{timestamp}.pdf"
        
        pdf_path = report_generator.generate_pdf_report(results, pdf_filename)
        print(f"PDF report generated: {pdf_path}")
        
        # Save JSON results
        json_filename = f"{repo_name}_results_{timestamp}.json"
        json_path = os.path.join(args.output_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"JSON results saved: {json_path}")
        
    finally:
        # Clean up temporary directory
        if repo_dir and "/tmp/" in repo_dir:
            shutil.rmtree(repo_dir)

if __name__ == "__main__":
    main()