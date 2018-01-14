class Emotion(object):
    def __init__(self, emotion):
        # ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
        # TODO: 2_Interact with emotions (jokes, give hug,..)
        #       Switch case/ if case to interact with emotions
        self.emotion = emotion

        if emotion == 'angry':
            print('angry')
        elif emotion == 'disgusted':
            print('disgusted')
        elif emotion == 'fearful':
            print('fearful')
        elif emotion == 'happy':
            print('happy')
        elif emotion == 'sad':
            print('sad')
        elif emotion == 'surprised':
            print('surprised')
        elif emotion == 'neutral':
            print('neutral')
        else:
            print('oeps...')
