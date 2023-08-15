"""
Object Tracking with OpenCV

In object detection, we identify an object in an image, draw a bounding box around it, and classify the object. 

Object tracking, on the other hand, is a discipline within computer vision that aims to follow moving objects across a sequence of video frames. These objects can include humans, animals, vehicles, or other interesting items like a soccer ball in a match.

Object tracking has numerous practical applications such as surveillance, medical imaging, traffic flow analysis, unmanned vehicles, people counting, audience flow analysis, and human-computer interaction.

Technically, object tracking starts with object detection, which involves identifying objects in an image and assigning bounding boxes to them.

An object tracking algorithm assigns an identity to each identified object in the image and then attempts to track this identity across subsequent frames, determining the new position of the same object as it moves.

Re-identification: Matching the same object across consecutive frames, as objects can move unpredictably in and out of the frame and need to be linked back to previously seen instances in the video.

Occlusion: Objects can become partially or fully occluded in certain frames due to the presence of other objects obstructing them, necessitating methods to handle such cases.

Identity Ambiguity: When two objects intersect, determining which is which becomes essential for accurate tracking.

Motion Blur: Objects can appear differently due to their own motion or camera movements, introducing challenges in maintaining consistent tracking.

Viewpoint Variations: Objects may appear drastically different from various viewpoints, demanding consistent identification across all perspectives.

Scale Changes: The scale of objects in a video can significantly alter, for instance, due to camera zooming.

Lighting Conditions: Changes in lighting throughout a video can profoundly impact how objects appear, making consistent detection more challenging.





"""