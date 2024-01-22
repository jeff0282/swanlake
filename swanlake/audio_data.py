
import      numpy as np
import      pydub

from        typing          import          Type
from        typing          import          BinaryIO

from        pathlib         import          Path


class BaseAudioData:
    """
    BaseAudioData

    Implements accessing and seeking through binary data stored in a 
    given BinaryIO File Object.
    """

    # CONSTANTS
    _FILEOBJ_SEEK_ABSOLUTE = 0
    _FILEOBJ_SEEK_RELATIVE = 1

    # TYPING
    _source:        BinaryIO
    _position:      int # frame number
    _framesize:     int # in bytes
    _channels:      int
    _samplewidth:   int # in bytes
    _framerate:     int
    _framecount:    int


    def __init__(self,
                 source:        BinaryIO,
                 *,
                 channels:      int,
                 samplewidth:   int,
                 framerate:     int,
                 framecount:    int
                 ) -> None:
        """
        Create a new BaseAudioStream.

        This should only really be used inside subclasses :)
        """

        self._source        = source
        self._position      = 0
        self._framesize     = (samplewidth * channels)
        self._channels      = channels
        self._samplewidth   = samplewidth
        self._framerate     = framerate
        self._framecount    = framecount


    #
    # Properties
    # 
    source: BinaryIO = property(lambda self: self._source)
    """A BinaryIO File Object pointing to the binary audio data"""

    position: int = property(lambda self: self._position)
    """The current frame currently seeked to"""

    framesize: int = property(lambda self: self._framesize)
    """The size of a single frame"""

    channels: int = property(lambda self: self._channels)
    """The number of channels present"""

    samplewidth: int = property(lambda self: self._samplewidth)
    """The size of each individual sample in bytes (NOT the size of each frame)"""

    bitdepth: int = property(lambda self: self.samplewidth * 8)
    """The size of each individual sample in bits (NOT the size of each frame)"""

    framerate: int = property(lambda self: self._framerate)
    """The amount of frames in each second"""

    framecount: int = property(lambda self: self._framecount)
    """The total number of frames"""

    datatype: np.dtype = property(lambda self: getattr(np, f"int{self.bitdepth}"))
    """The datatype used for each sample when requesting frames"""


    #
    # High-Level Methods
    #
    def seek(self, frame_offset: int) -> None:
        """
        Move position from current position by a given offset
        """

        self._move_fileobj_cursor(frame_offset, self._FILEOBJ_SEEK_RELATIVE)


    def jump(self, frame_offset: int) -> None:
        """
        Move position to a specific frame

        Raises ValueError if given 
        """

        self._move_fileobj_cursor(frame_offset, self._FILEOBJ_SEEK_ABSOLUTE)


    def read(self, frames: int) -> Type[np.array]:
        """
        Read a number of frames from the current position to the right. 
        
        Length of return array will be less than requested frames if insufficient frames to fill.
        """

        # get chunk of data and transform into numpy array
        return self._create_frame_array(self._get_chunk_from_infront(frames))
        


    def read_left(self, frames: int) -> Type[np.array]:
        """
        Read a number of frames from the current position from the left of the current position

        Does NOT read in reverse, simply just returns the chunk before the current position 
        """

        return self._create_frame_array(self._get_chunk_from_behind(frames))


    #
    # Low-Level Methods
    #
    def _create_frame_array(self, byte_data: bytes) -> Type[np.array]:
        """
        Create a dimensional array from the given byte data in shape (<length>, channels)
        """
        
        # create a dimensional array with the given byte data,
        # use known data about audio to form correctly
        return np.ndarray(shape     = (len(byte_data // self.framesize), self.channels), 
                          channels  = self.channels, 
                          dtype     = self.datatype,
                          buffer    = byte_data)
    

    def _get_chunk_from_infront(self, chunk_size: int) -> bytes:
        """
        Get a chunk of bytes from audio data, reading from infront of current position

        Params:
        `chunk_size: int` - The size of the chunk to retrieve in frames
        """

        return self.source.read(chunk_size)


    def _get_chunk_from_behind(self, chunk_size: int) -> bytes:
        """
        Get a chunk of bytes from audio data, reading from behind current position

        Does NOT return the bytes in reverse order. Order is as in the original data

        Params:
        `chunk_size: int` - The size of the chunk to retrieve in frames
        """

        # Reading backwards is unfortunately not very nice
        # ---
        # Standard practice would be to seek backwards, and check length of read(). 
        # However, seeking below start (zero) can result in OSError, so we must be defensive
        # Fucking WATCH this ;)

        # if we would end up seeking below 0, start from frame 0 and 
        # adjust the chunksize by the difference to avoid overshooting current frame
        starting_frame  = self.position - chunk_size
        if starting_frame < 0:
            chunk_size = chunk_size - starting_frame
            starting_frame = 0

        # we need to temporarily seek to the starting frame,
        # without changing the current position
        self._move_fileobj_cursor(starting_frame, self._FILEOBJ_SEEK_ABSOLUTE)
        chunk = self.source.read(chunk_size)

        # seek back to position and return chunk data
        self._move_fileobj_cursor(self.position, self._FILEOBJ_SEEK_ABSOLUTE)
        return chunk


    def _move_position(self, frame_offset: int, mode: int = _FILEOBJ_SEEK_ABSOLUTE) -> None:
        """
        Move the current position and fileobj cursor to a given frame

        Params:
        - `frame_offset: int` - The number of frames to move
        - [OPTIONAL, DEFAULT = 0] `mode: int`   
            The `_whence` arg in <fileobject>.seek(): where
            - 0 -> absolute (from start, like and array index)
            - 1 -> relative to current cursor position
            - 2 -> relative to end of the file 

        Raises:
        - `ValueError`: if frame_offset results in a new position outwith the bounds of the audio data
        """

        try:
            self._move_fileobj_cursor(frame_offset, mode)

        except Exception as e:
            self._position = (self.source.tell() // self.framesize)


    def _move_fileobj_cursor(self, frame_offset: int, mode: int = _FILEOBJ_SEEK_ABSOLUTE) -> None:
        """
        Move the internal file objects cursor to a given frame_offset

        Does NOT change the current position, only the fileobj cursor

        Params:
        - `frame_offset: int` - The number of frames to move
        - [OPTIONAL, DEFAULT = 0] `mode: int`   
            The `_whence` arg in <fileobject>.seek(): where
            - 0 -> absolute (from start, like and array index)
            - 1 -> relative to current cursor position
            - 2 -> relative to end of the file 

        Raises:
        - `ValueError`: if frame_offset results in a new position outwith the bounds of the audio data
        """

        # calculate the byte position of the frame offset
        new_pos = (frame_offset * self.framesize)

        # check that the frame position to jump to is within bounds
        if new_pos < 0 or new_pos >= self.framecount:
            ValueError(f"Cannot seek to a frame {new_pos} (given frame offset {frame_offset}): Out of Bounds")

        # if no error, jump !!!
        self.source.seek(new_pos, mode)