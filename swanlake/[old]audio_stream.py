
import      math
import      numpy as np
import      sounddevice as sd

from        typing      import      Any
from        typing      import      Type

from        .           import      AudioData



class AudioStream:
    """
    
    """

    # CONSTANTS
    DEFAULT_BLOCK_SIZE  = 1024 # in frames
    MODE_STOPPED        = 0
    MODE_PAUSED         = 1
    MODE_PLAYING        = 2


    # TYPING
    # playback properties
    curframe:   int     # index of current frame in AudioData   
    _loop:      bool    # True to loop
    _direction: int     # 1 == forward, -1 == reverse

    # audio and streaming properties
    _mode:      int # see MODE_* constants
    _stream:    Type[sd.OutputStream]
    _audio:     Type[AudioData]


    def __init__(self, 
                 audiodata:     Type[AudioData],
                 *,
                 loop:          bool            = False,
                 reverse:       bool            = False,
                 fr_override:   int | None      = None
                 ) -> None:
        """
        Create a new AudioStream using an 
        """

        # setup stream
        self._audio     = audiodata
        framerate       = fr_override if fr_override else self._audio.framerate
        self._stream    = sd.OutputStream(samplerate        = framerate,
                                          blocksize         = self.DEFAULT_BLOCK_SIZE,
                                          channels          = self._audio.channels,
                                          callback          = self._next_block_callback,
                                          finished_callback = self._stop_playback_callback,
                                          dtype             = audiodata.dtype) 
        
        # set playback and streaming state
        self._set_default_playback_state()
        self.loop       = loop
        self.reverse    = reverse


    @property
    def framerate(self) -> int:
        """
        The playback framerate
        """

        return self._stream.samplerate


    @property
    def audio(self) -> Type[AudioData]:
        """
        The AudioData instance this stream is set to play
        """

        return self._audio
        

    @property
    def loop(self) -> bool:
        """
        Whether or not this stream is set to loop
        """

        return self._loop
    

    @loop.setter
    def loop(self, new_val: bool) -> None:
        """
        Set whether or not this stream should loop

        True to loop, False otherwise
        """

        self._loop = new_val


    @property
    def reverse(self) -> bool:
        """
        Whether or not this loop is playing in reverse
        """

        return True if self.direction == -1 else False
    

    @reverse.setter
    def reverse(self, new_val: bool) -> None:
        """
        Set whether or not this stream should play in reverse

        True for reverse, False otherwise
        """

        # 1 == forward, -1 == reverse
        self.direction = -1 if new_val else 1


    @property
    def direction(self) -> int:
        """
        The direction this stream is playing, -1 for reverse, 1 for forwards
        """

        return self._direction
    

    @direction.setter
    def direction(self, new_val: int) -> None:
        """
        Set the direction this stream should play.

        -1 for reverse, 1 for forwards
        """

        if new_val != 1 and new_val != -1:
            raise ValueError(f"Invalid Direction: -1 and 1 are valid. Given '{new_val}'")

        self._direction = new_val


    def stop(self) -> None:
        """
        Stop all playback and reset AudioStream to beginning
        """

        if self._stream.active:
            self._mode = self.MODE_STOPPED
            self._stream.stop()


    def pause(self) -> None:
        """
        Pause playback at current position
        """

        if self._stream.active:
            self._mode = self.MODE_PAUSED
            self._stream.stop()


    def play(self) -> None:
        """
        Start/Resume playback at current position

        If playback is stopped and set to play in reverse, will
        jump to the end of the audio before playing
        """

        # HACK !!!!
        # sounddevice.OutputStream
        # When playback is stopped due to reaching end of audio,
        # the OutputStream will no longer read data in from the callback when requested to start
        # (this is documented behaviour: the callback will not be called when CallbackStop() or 
        # CallbackAbort() are raised)
        # This, for whatever reason, fixes that :)
        if self._mode == self.MODE_STOPPED:
            self._stream.abort()

        self._mode = self.MODE_PLAYING
        if self._stream.stopped:
            if self.reverse:
                self.jump_to_end()
            self._stream.start()


    def jump(self, seconds: int) -> None:
        """
        Jump to a specific time in the audiodata
        """

        self.jump_to_frame(math.floor(seconds * self.framerate))


    def jump_to_frame(self, frame: int) -> None:
        """
        Jump to a specific frame in the audio
        """

        if frame < 0 or frame > self.audio.framecount:
            raise ValueError("Cannot jump to frame outwith length of audio")
        
        self.curframe = frame


    def jump_to_start(self) -> None:
        """
        Jump to the start of the audio
        """

        self.jump_to_frame(0)


    def jump_to_end(self) -> None:
        """
        Jump to the end of the audio
        """

        self.jump_to_frame(len(self._audio) - 1)


    def _stop_playback_callback(self) -> None:
        """
        `sounddevice.OutputStream` finish callback

        This is called when .stop() or .abort() is called on the 
        internal `sounddevice.OutputStream` (when stopping and pausing playback)

        NOTE: Playback mode MUST be set BEFORE this is called 
        (i.e. before CallbackStop in callback, before called OutputStream.stop()/.abort())

        If called with mode set to MODE_STOPPED or MODE_PLAYING, will reset streaming state
        If called with mode set to MODE_PAUSED, will preserve the streaming state
        """

        # this will be called when pausing, so checking for MODE_PAUSED 
        if self._mode != self.MODE_PAUSED:
            self._set_default_playback_state()


    def _set_default_playback_state(self) -> None:
        """
        Set the playback defaults (starting frame, playback mode, etc)

        Does NOT stop playback, etc. Simply resets internal vars
        """

        self.jump_to_start()
        self._mode = self.MODE_STOPPED


    def _set_default_playback_props(self) -> None:
        """
        Set the playback properties (looping, reverse, etc) to their defaults
        """

        self.loop       = False
        self.reverse    = False
     

    def _next_block_callback(self,
                             outdata:    Type[np.ndarray],
                             frames:     int,
                             time:       Any, # CDATA
                             status:     sd.CallbackFlags
                             ) -> None:
        """
        Sounddevice.stream callback

        Fills the buffer `outdata` with the data of the next block of data.
        If insufficient data to fill `outdata, fills with zeros

        Returns a slice or the input array, and the index of the last frame
        """

        def get_next_block(frame_array:   Type[np.array],
                           direction:     int,
                           start_frame:   int,
                           slice_size:    int
                           ) -> tuple[Type[np.array], int]:

            is_within_bounds = lambda indx: True if 0 <= indx and indx < len(frame_array) else False
            
            # Get boundaries and check if within bounds
            # Put boundaries in correct order (based on direction)
            # (block boundaries in form [Given Frame, Is Within Bounds])
            start       = start_frame
            raw_end     = start + (slice_size * direction)
            end         = raw_end if is_within_bounds(raw_end) else None

            # Calculate frame index of the last frame in new block
            # if end falls outwith bounds, the endframe should be equal to:
            # - 0 if direction == -1
            # - len(frame_array) if direction == 1
            end_frame_idx = raw_end if raw_end else len(frame_array) if direction == 1 else 0

            return frame_array[start : end : direction], end_frame_idx



        if status.output_underflow:
            print("WARN: OUTPUT UNDERFLOWING")

        block_data, end_block_idx = get_next_block(self._audio.data, self._direction, self.curframe, self.DEFAULT_BLOCK_SIZE)

        if len(block_data) < len(outdata):
            outdata[:len(block_data)] = block_data
            outdata[len(block_data):].fill(0)
            if self.loop:
                self.jump_to_end() if self.reverse else self.jump_to_start()
            else:
                raise sd.CallbackStop()
            
        else:
            outdata[:] = block_data
            self.curframe = end_block_idx
        