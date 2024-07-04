from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Img2VideoInputMsg(_message.Message):
    __slots__ = ("conditionImages", "ipadapterImage", "referImage", "prompt", "scaleRatio", "eyeBlinksFactor", "dest", "fps", "length", "motionSpeed", "batch")
    CONDITIONIMAGES_FIELD_NUMBER: _ClassVar[int]
    IPADAPTERIMAGE_FIELD_NUMBER: _ClassVar[int]
    REFERIMAGE_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    SCALERATIO_FIELD_NUMBER: _ClassVar[int]
    EYEBLINKSFACTOR_FIELD_NUMBER: _ClassVar[int]
    DEST_FIELD_NUMBER: _ClassVar[int]
    FPS_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    MOTIONSPEED_FIELD_NUMBER: _ClassVar[int]
    BATCH_FIELD_NUMBER: _ClassVar[int]
    conditionImages: str
    ipadapterImage: str
    referImage: str
    prompt: str
    scaleRatio: float
    eyeBlinksFactor: float
    dest: str
    fps: int
    length: int
    motionSpeed: float
    batch: int
    def __init__(self, conditionImages: _Optional[str] = ..., ipadapterImage: _Optional[str] = ..., referImage: _Optional[str] = ..., prompt: _Optional[str] = ..., scaleRatio: _Optional[float] = ..., eyeBlinksFactor: _Optional[float] = ..., dest: _Optional[str] = ..., fps: _Optional[int] = ..., length: _Optional[int] = ..., motionSpeed: _Optional[float] = ..., batch: _Optional[int] = ...) -> None: ...

class Img2VideoAsyncInputMsg(_message.Message):
    __slots__ = ("input", "progressCallback", "finishCallback")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    PROGRESSCALLBACK_FIELD_NUMBER: _ClassVar[int]
    FINISHCALLBACK_FIELD_NUMBER: _ClassVar[int]
    input: Img2VideoInputMsg
    progressCallback: str
    finishCallback: str
    def __init__(self, input: _Optional[_Union[Img2VideoInputMsg, _Mapping]] = ..., progressCallback: _Optional[str] = ..., finishCallback: _Optional[str] = ...) -> None: ...

class Img2VideoResponse(_message.Message):
    __slots__ = ("result", "dest")
    RESULT_FIELD_NUMBER: _ClassVar[int]
    DEST_FIELD_NUMBER: _ClassVar[int]
    result: bool
    dest: str
    def __init__(self, result: bool = ..., dest: _Optional[str] = ...) -> None: ...
