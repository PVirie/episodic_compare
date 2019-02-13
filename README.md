# episodic_compare
Learning to compare with episodic memory

Catastrophic forgetting happens because we train one model on many sequential tasks, what if there are not several tasks but one? That task is matching! 

Humans have their sensors fixed to the brain, and each sensor pick up the same type of signal over the course of life. Processing the same signal over and over is the only task they do. Hence could we let the conventional mini-batch learning handle the generalization of this task, and let another part handles actual tasks to which we human refer.

Task difference comes from the abstract notion of human application, not in the information space itself. The lower level processes somehow infer the most probable high level states to be consumed by higher level programs, procedures or routines specific to a given application. These programs should be learned via simple one-shot demonstration, the way many people mistake for the ability to quickly learn without extensive training.

Therefore I combine these two features into one model using learning to compare with episodic memory, to demonstrate the seemingly magical ability of humans to continual learn without having to suffer from catastrophic forgetting. Or more likely, we still do suffer from catastrophic forgetting, but on the ability to generalize match, not the task of matching itself.



