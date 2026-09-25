# GPT grading prompts from DriveBench (https://github.com/drive-bench/toolkit), Apache-2.0, see LICENSE.
PERCEPTION_MCQ_PROMPT = """Please evaluate the multiple-choice answer on a scale from 0 to 100, where a higher score reflects precise alignment with the correct answer and well-supported reasoning. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

1. Answer Correctness (50 points):
- Exact Match (50 points): Assign 50 points if the predicted answer exactly matches the correct answer.
- No Match (0 points): Assign 0 points if the predicted answer does not match the correct answer, regardless of explanation quality.
        
2. Object Recognition (10 points):
- Award up to 5 points for accurately identifying all relevant object(s) in the scene.
- Award up to 5 points for correct descriptions of the identified object(s), including attributes like colors, materials, sizes, or shapes.
- Guideline: Deduct points for any missing, misidentified, or irrelevant objects, particularly if they are crucial to the driving context. Deduct points if any important visual details are missing, incorrect, or overly generalized, especially if they affect comprehension or recognition.
        
3. Object Location and Orientation (15 points):
- Score up to 5 points for precise description of the object's location, orientation, or position relative to the ego vehicle.
- Award up to 5 points for acknowledging environmental factors, such as lighting, visibility, and other conditions that influence perception.
- Score up to 5 points based on how well the answer reflects an understanding of situational context, such as obstacles, traffic flow, or potential hazards.
- Guideline: Deduct points for inaccuracies or omissions in spatial information that could affect scene understanding. Deduct points if the answer fails to consider factors impacting object visibility or situational awareness. Deduct points for overlooked or misinterpreted contextual factors that may impact driving decisions.

4. Environmental Condition Awareness (15 points):
- Award up to 15 points if the explanation considers environmental conditions (e.g., weather or sensor limitations) that could impact perception.
- Guideline: Deduct points if relevant environmental conditions are ignored or inadequately addressed.

5. Clarity of Reasoning (10 points):
- Award up to 5 points for clear, logically structured reasoning that is easy to understand.
- Assign up to 5 points for grammatical accuracy and coherent structure.
- Guideline: Deduct points for vague or confusing explanations that hinder comprehension. Deduct points for grammar or syntax issues that impact clarity or logical flow.

Assign 0 points from criteria 2 to 5 if no explanation is provided.

Here is the multiple-choice question: "{QUESTION}"

Here is the ground truth object visual description: "{DESC}"

Here is the correct answer: "{GT}"

Here is the predicted answer and explanation (if any): "{PRED}"

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Answer Correctness (50 points): 
2. Object Recognition (10 points):
3. Object Location and Orientation (15 points):
4. Environmental Condition Awareness (15 points):
5. Clarity of Reasoning (10 points): 
Total Score: 

Brief Summary: """


PERCEPTION_VQA_PROMPT = """Please evaluate the predicted answer on a scale from 0 to 100, where a higher score reflects precise alignment with the correct answer and well-supported reasoning. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

1. Action Alignment (20 points):
- Assign up to 20 points based on how accurately the predicted action (e.g., forward, turn left, turn right) matches the correct answer.
- Guideline: Award full points only for exact matches or highly similar actions. Deduct points for any inaccuracies or missing elements. Assign 0 points if no action prediction is provided.

2. Motion Precision (20 points):
- Award up to 20 points based on how closely the predicted motion (e.g., speed up, decelerate) aligns with the correct motion in the answer.
- Guideline: Deduct points if the predicted motion fails to match the type or intensity of the correct answer. Ensure that the intended speed or deceleration aligns accurately with the driving context. Assign 0 points if no motion prediction is provided.
  
3. Driving Context Appropriateness (15 points):
- Score up to 15 points for the relevance of the predicted answer to the driving context implied by the correct answer, emphasizing logical alignment with the situation.
- Guideline: Award higher scores only if the answer fully reflects an accurate understanding of the driving context. Deduct points if the action or motion is illogical or does not align with the scenario's requirements.

4. Situational Awareness (15 points):
- Award up to 15 points for demonstrated awareness of environmental factors (e.g., traffic participants, obstacles) relevant to the action or motion.
- Guideline: Deduct points if the answer misses key situational details that may lead to unsafe or incorrect predictions.

5. Conciseness and Clarity (20 points):
- Assess the clarity and brevity of the predicted answer. Answers should be concise, clear, and easy to understand, effectively communicating the intended actions and motions.
- Guideline: Deduct points for verbosity, ambiguity, or lack of focus that could hinder quick comprehension.

6. Grammar (10 points):
- Evaluate the grammatical accuracy and structure of the answer. Assign up to 5 points for clarity and logical flow, and up to 5 points for grammatical accuracy.
- Guideline: Deduct points for grammar or syntax issues that reduce readability or coherence.

Here is the multiple-choice question: "{QUESTION}"

Here is the predicted answer: "{PRED}"

Here is the correct answer: "{GT}"

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Action Alignment (20 points): 
2. Motion Precision (20 points):
3. Driving Context Appropriateness (15 points):
4. Situational Awareness (15 points):
5. Conciseness and Clarity (20 points): 
6. Grammar (10 points):
Total Score: 

Brief Summary: """


PREDICTION_MCQ_PROMPT = """Please evaluate the predicted answer for the Yes/No question on a scale from 0 to 100, where a higher score reflects precise alignment with the correct answer and a well-supported explanation. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

1. Answer Correctness (40 points):
- Exact Match (40 points): Assign 40 points if the predicted Yes/No answer exactly matches the correct answer.
- No Match (0 points): Assign 0 points if the predicted answer does not match the correct answer, regardless of explanation quality.
        
2. Object Category Identification (15 points):
- Score up to 15 points for accurately identifying the object's category.
- Guideline: Deduct points for any inaccuracies or missing elements in the category identification, particularly if they affect understanding or recognition of the object’s role in the scene.
        
3. Object Visual Appearance (15 points):
- Score up to 15 points for an accurate description of the object's visual appearance (e.g., colors, materials, size, or shape) as relevant to the question.
- Guideline: Deduct points if any important visual details are missing, incorrect, or overly generalized, especially if they impact the explanation or perception of the object’s function.

4. Object Position and Motion (15 points):
- Score up to 15 points for correctly identifying the object's location, orientation, and motion (if applicable) relative to the ego vehicle.
- Guideline: Deduct points for inaccuracies in spatial information, positioning, or motion. Include deductions if relevant motion or orientation details are omitted or incorrect.

5. Explanation Clarity and Justification (15 points):
- Score up to 15 points for the clarity, logical structure, and justification of the explanation provided.
- Guideline: Deduct points for vague, confusing, or insufficient explanations that fail to justify the Yes/No answer clearly and logically.

Assign 0 points from criteria 2 to 5 if no explanation is provided.

Here is the multiple-choice question: "{QUESTION}"

Here is the correct answer: "{GT}"

Here is the predicted answer and explanation (if any): "{PRED}"

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Answer Correctness (40 points):
2. Object Category Identification (15 points):
3. Object Visual Appearance (15 points):
4. Object Position and Motion (15 points):
5. Explanation Clarity and Justification (15 points):
Total Score: 

Brief Summary: """


PREDICTION_VQA_PROMPT = """Please evaluate the predicted answer on a scale from 0 to 100, where a higher score reflects precise alignment with the correct answer and well-supported reasoning. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

1. Object Identification and Priority Order (20 points):
- Score up to 20 points for accurately identifying the correct objects in the correct priority order (e.g., first, second, third) as indicated in the question.
- Guideline: Deduct points for any missed, misidentified, or out-of-sequence objects, particularly if this affects the logic of the driving prediction.

2. Object Category and Visual Description Accuracy (20 points):
- Score up to 20 points for accurately describing each object’s category (e.g., traffic sign, vehicle) and relevant visual attributes (e.g., color, type, size) as necessary for the scene.
- Guideline: Deduct points for incorrect or overly generalized descriptions of categories or visual attributes, especially if they impact scene comprehension or recognition.
  
3. State of the Object (15 points):
- Score up to 15 points based on the accuracy of the object’s state (e.g., moving, stationary) and alignment with the correct answer.
- Guideline: Deduct points if the predicted state does not match the correct state or if critical details of the object’s status are omitted.

4. Recommended Action for Ego Vehicle (15 points):
- Score up to 15 points for accurately identifying the action the ego vehicle should take in response to each object (e.g., continue ahead, slow down).
- Guideline: Deduct points for actions that are inappropriate, unclear, or lacking necessary detail, especially if they contradict the driving context.

5. Logical Flow and Reasonableness of Prediction (20 points):
- Score up to 20 points for the logical consistency and reasonableness of the entire response. The answer should reflect a clear, step-by-step rationale that aligns logically with the question’s driving context.
- Guideline: Deduct points for inconsistencies, contradictions, or illogical reasoning that undermine the reliability of the prediction.

6. Clarity and Grammar (10 points):
- Score up to 10 points for clarity, coherence, and grammatical accuracy. Assign up to 5 points for logical structure and readability, and up to 5 points for grammatical accuracy.
- Guideline: Deduct points for ambiguous explanations, unclear reasoning, or grammatical errors that reduce clarity.

Here is the question: "{QUESTION}"

Here is the correct answer: "{GT}"

Here is the predicted answer: "{PRED}"

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Object Identification and Priority Order (20 points):
2. Object Category and Visual Description Accuracy (20 points):
3. State of the Object (15 points):
4. Recommended Action for Ego Vehicle (15 points):
5. Logical Flow and Reasonableness of Prediction (20 points):
6. Clarity and Grammar (10 points):
Total Score: 

Brief Summary: """


PLANNING_VQA_PROMPT = """Please evaluate the predicted answer on a scale from 0 to 100, where a higher score reflects precise alignment with the correct answer and well-supported reasoning. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

1. Action Prediction Accuracy (40 points):
- Score up to 40 points based on the accuracy of the predicted action for the ego vehicle (e.g., keep going, turn left, turn right, accelerate, decelerate) in response to the contextual information.
- Guideline: Award full points only for exact or highly similar action matches. Deduct points for inaccuracies or actions that do not match the correct answer, especially if they could compromise safety or appropriateness in context.

2. Reasoning and Justification (20 points):
- Score up to 20 points for a clear and logical explanation of why the action is chosen, ensuring that the reason aligns with safety, environmental factors, or other relevant considerations.
- Guideline: Deduct points if the reasoning lacks clarity, omits relevant details, or includes contradictions. The explanation should justify the action in a way that is suitable for the scenario provided.
  
3. Probability or Confidence Level (15 points):
- Score up to 15 points for accurately predicting the probability or confidence level of the action being safe or feasible (e.g., high probability if no obstacles are present).
- Guideline: Deduct points if the probability level is missing, implausible, or does not align with the action or reasoning provided.

4. Contextual Awareness and Safety Considerations (15 points):
- Score up to 15 points for reflecting an awareness of the driving context, including potential obstacles, traffic participants, and safety implications.
- Guideline: Deduct points for failing to consider contextual factors that may impact the ego vehicle's decision, especially if they could lead to unsafe actions.

5. Conciseness and Clarity (10 points):
- Assess the clarity and brevity of the answer. Answers should be concise and easy to understand, effectively communicating the intended actions and rationale.
- Guideline: Deduct points for verbosity, ambiguity, or lack of focus that could hinder quick comprehension. Assign 0 points if no explanation is provided.

Here is the question: "{QUESTION}"

Here is the ground truth object visual description: "{DESC}"

Here is the correct answer: "{GT}"

Here is the predicted answer: "{PRED}"

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Action Prediction Accuracy (40 points):
2. Reasoning and Justification (20 points):
3. Probability or Confidence Level (15 points):
4. Contextual Awareness and Safety Considerations (15 points):
5. Conciseness and Clarity (10 points):
Total Score: 

Brief Summary: """


BEHAVIOR_MCQ_PROMPT = """Please evaluate the predicted answer on a scale from 0 to 100, where a higher score reflects precise alignment with the correct answer and well-supported reasoning. Be strict and conservative in scoring, awarding full points only when all criteria are fully met without error. Deduct points for minor inaccuracies, omissions, or lack of clarity. Distribute the Total Score across the following criteria:

1. Answer Correctness (50 points):
- Exact Match (50 points): Assign 50 points if the predicted answer exactly matches the correct answer from the options.
- No Match (0 points): Assign 0 points if the predicted answer does not match the correct answer, regardless of explanation quality.

2. Behavioral Understanding and Detail (15 points):
- Score up to 15 points for accurately capturing the behavior details of the ego vehicle (e.g., going straight, steering left, driving speed) as outlined in the correct answer.
- Guideline: Deduct points if the explanation misses key behavioral aspects (e.g., direction, speed) that are essential to understanding the ego vehicle's movement.
  
3. Reasoning and Justification (15 points):
- Score up to 15 points for a clear and logical explanation justifying the chosen answer. The explanation should accurately describe why the selected behavior is appropriate, considering factors such as road direction, traffic flow, or other environmental clues.
- Guideline: Deduct points if the reasoning lacks clarity, includes irrelevant details, or contradicts the behavior of the ego vehicle as described in the correct answer.

4. Contextual Relevance (10 points):
- Score up to 10 points for the relevance of the explanation to the driving context, ensuring that it considers any environmental or situational factors that may influence the ego vehicle’s behavior.
- Guideline: Deduct points if the explanation fails to consider context, such as road conditions or nearby objects, that might impact the vehicle’s behavior.

5. Clarity and Grammar (10 points):
- Score up to 10 points for clarity, coherence, and grammatical accuracy of the explanation. The response should be concise and easy to understand.
- Guideline: Deduct points for confusing language, vague statements, or grammatical errors that hinder comprehension.

Assign 0 points from criteria 2 to 5 if no explanation is provided.

Here is the question: "{QUESTION}"

Here is the correct answer: "{GT}"

Here is the predicted answer: "{PRED}"

Please fill in the following scoring sheet, and then provide a brief summary supporting the score:
1. Answer Correctness (50 points):
2. Behavioral Understanding and Detail (15 points):
3. Reasoning and Justification (15 points):
4. Contextual Relevance (10 points):
5. Clarity and Grammar (10 points):
Total Score: 

Brief Summary: """
