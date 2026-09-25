framewise_message = """You are provided with a list of frames and their corresponding sampled indices as follows:
{frame_info}
Described each of the frames in one sentence. The description should be only based on the frame. Return your output as frame_<frame_index>:<description>."""

chunked_video_message = """ You are captioning a video that is provided to you in consecutive parts.

Previous description:
{previous_outputs}

Now describe the next part of the video.

Your response will be concatenated directly after the previous description to form the final description of the entire video. Therefore, continue from the previous description rather than rewriting or summarizing it.

Describe only the new visual information and events present in the current video part. Avoid repeating anything already mentioned in the previous description. Make the continuation read naturally when appended to the previous text.

Be concise and factual. Do not include timestamps, segment labels, bullet points, or commentary about the captioning process. Do not mention that the video was split into parts.

Output only the new continuation of the description.
"""

independent_frame_message = "Provide a detailed description of this frame in one or two sentences."

generic_video_message = "Provide a detailed and concise description of this video. Do not repeat any details"
