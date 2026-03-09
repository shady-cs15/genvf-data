import re
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class Thought:
    """Represents an individual thought in a reasoning trajectory."""
    content: str
    start_idx: int
    end_idx: int
    transition_keyword: str = None


class TrajectorySegmenter:
    """Segments reasoning trajectories into individual thoughts using transition keywords."""
    
    def __init__(self):
        # Define transition keywords
        self.transition_keywords = [
            'But', 'Wait', 'Alternatively', 'However', 'Hmm', 'Hmmm', 
            'Not sure', 'Going back', 'Backtrack', 'Trace back', 'Another'
        ]

        # self.transition_keywords = [
        #     'But', 'Wait', 'Alternatively', 'However', 'Hmm', 'Hmmm', 
        #     'Backtrack', 'Another'
        # ]
    
    def segment_trajectory(self, text: str) -> List[Thought]:
        """
        Segment a trajectory into individual thoughts based on transition keywords.
        
        Args:
            text: The full reasoning trajectory text
            
        Returns:
            List of Thought objects representing individual thoughts
        """
        thoughts = []
        
        # Find all transition keyword positions
        transitions = self._find_transitions(text)
        
        if not transitions:
            # If no transitions found, treat entire text as one thought
            return [Thought(
                content=text.strip(),
                start_idx=0,
                end_idx=len(text)
            )]
        
        # Create thoughts based on transitions
        start_idx = 0
        
        for i, (keyword, keyword_pos, sentence_start) in enumerate(transitions):
            # Add thought before this transition (if not the first one)
            if i == 0:
                # First thought: from <|im_start|>assistant 到第一个 transition
                assistant_pos = text.find("<|im_start|>assistant")
                if assistant_pos != -1:
                    start_idx = assistant_pos
                if sentence_start > start_idx:
                    content = text[start_idx:sentence_start].strip()
                    if content:
                        thoughts.append(Thought(
                            content=content,
                            start_idx=start_idx,
                            end_idx=sentence_start,
                            transition_keyword=None
                        ))
            
            # Set start for next thought (from this transition)
            start_idx = sentence_start
            
            # If this is not the last transition, end at next transition
            if i < len(transitions) - 1:
                next_sentence_start = transitions[i + 1][2]
                content = text[start_idx:next_sentence_start].strip()
                if content:
                    thoughts.append(Thought(
                        content=content,
                        start_idx=start_idx,
                        end_idx=next_sentence_start,
                        transition_keyword=keyword
                    ))
                start_idx = next_sentence_start
        
        # Add final thought from last transition to end
        if transitions:
            final_content = text[start_idx:].strip()
            if final_content:
                thoughts.append(Thought(
                    content=final_content,
                    start_idx=start_idx,
                    end_idx=len(text),
                    transition_keyword=transitions[-1][0]
                ))
        
        return thoughts
    
    def _find_transitions(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Find all transition keyword positions in the text.
        
        Returns:
            List of tuples: (keyword, keyword_position, paragraph_start_position)
        """
        transitions = []
        
        # Split into paragraphs for better boundary detection
        paragraphs = re.split(r'\n\s*\n', text)
        current_pos = 0
        
        for paragraph in paragraphs:
            paragraph_start = current_pos
            paragraph = paragraph.strip()
            
            if not paragraph:
                current_pos += len(paragraph) + 2  # +2 for \n\n
                continue
            
            # Look for transition keywords at the start of paragraphs
            for keyword in self.transition_keywords:
                # Create pattern that matches keyword at paragraph start
                pattern = rf'\b{re.escape(keyword)}\b'
                match = re.search(pattern, paragraph, re.IGNORECASE)
                
                if match and match.start() <= 20:  # Keyword appears near paragraph start
                    keyword_pos = paragraph_start + match.start()
                    transitions.append((keyword, keyword_pos, paragraph_start))
                    break
            
            current_pos = text.find(paragraph, current_pos) + len(paragraph)
            # Find the actual end position including whitespace
            while current_pos < len(text) and text[current_pos] in '\n \t':
                current_pos += 1
        
        return transitions
    
    def analyze_trajectory(self, text: str) -> Dict:
        """
        Perform complete analysis of a reasoning trajectory.
        
        Args:
            text: The full reasoning trajectory text
            
        Returns:
            Dictionary with analysis results
        """
        thoughts = self.segment_trajectory(text)
        
        # Calculate average thought length
        avg_length = sum(len(t.content) for t in thoughts) / len(thoughts) if thoughts else 0
        
        return {
            'total_thoughts': len(thoughts),
            'thoughts': thoughts,
            'average_thought_length': avg_length,
            'transition_keywords_used': [t.transition_keyword for t in thoughts if t.transition_keyword],
            'trajectory_length': len(text)
        }
    
    def get_num_thoughts_and_steps_per_thought(self, analysis: Dict):
        """return number of thought blocks in a response, and its avg step per thought"""
        step_per_thougths = [
            len(re.split(r'\n\s*\n', thought.content))
            for thought in analysis['thoughts']
        ]
        avg_steps_per_thought = sum(step_per_thougths) / len(step_per_thougths)
        return analysis['total_thoughts'], avg_steps_per_thought



if __name__ == "__main__":
    # ==== input here    
    dataset_name = "haoranli-ml/GVF-3k"
    split = "train"
    response_column_name = "responses"
    # ==============
    
    segmenter = TrajectorySegmenter()
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.shuffle(seed=2026)
    item_list = []
    for item in dataset:
        for idx, resp in enumerate(item[response_column_name]):
            thoughts = segmenter.segment_trajectory(resp)
            thoughts = [t.content for t in thoughts]
            index = random.randint(1, len(thoughts) - 1)
            prefix = "\n\n".join(thoughts[:index])

            item_list.append(
                {
                    "problem": item["problem"], 
                    "source": item["source"],
                    "rubrics": item["rubrics"] if "rubrics" in item else None,
                    "answer": item["answer"] if "answer" in item else None,
                    "orignial_question_mean_reward": item["mean_reward"],
                    "prefix": prefix,
                    "prefix_end_index": index,
                    "full_response": resp, 
                }
            )
