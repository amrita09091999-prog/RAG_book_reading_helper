import json

class Process:
    def normalize_text(self,text):
        return " ".join(text.split())

    def extract_between(self,text, start, end=None):
        if start not in text:
            return None
        part = text.split(start, 1)[1]
        if end and end in part:
            part = part.split(end, 1)[0]
        return self.normalize_text(part)
    
    def extract_metric_block(self,text, metric_name):
        if metric_name not in text:
            return None

        block = text.split(metric_name, 1)[1]

        # Try to isolate until next metric
        for next_metric in ["Answer Relevance", "Groundness", "Retrieval Relevance"]:
            if next_metric != metric_name and next_metric in block:
                block = block.split(next_metric, 1)[0]

        block = self.normalize_text(block)

        # Extract score
        score = None
        if '"score"' in block:
            try:
                score = int(block.split('"score"', 1)[1].split(":", 1)[1].split(",", 1)[0])
            except:
                pass

        # Extract explanation
        explanation = None
        if '"Explanation"' in block:
            explanation = block.split('"Explanation"', 1)[1]
            explanation = explanation.split('"', 2)[1]

        return {
            "score": score,
            "explanation": explanation
        }
    
    def llm_output_to_json(self,raw_text):
        return {
            "User Query": self.extract_between(raw_text, "User Query -", "AI Response -"),
            "AI Response": self.extract_between(raw_text, "AI Response -", "Answer Relevance -"),
            "Metrics": {
                "Answer Relevance": self.extract_metric_block(raw_text, "Answer Relevance"),
                "Groundness": self.extract_metric_block(raw_text, "Groundness"),
                "Retrieval Relevance": self.extract_metric_block(raw_text, "Retrieval Relevance"),
            }
        }