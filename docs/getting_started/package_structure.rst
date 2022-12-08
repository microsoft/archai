Package Structure
=================

.. mermaid::

    graph LR
        log{ILog} --> std[StandardLogger]
        log --> l0[ILogMessageConsumer]
        log --> l1[ILogMessageConsumer]
        c0(audio.mp3.encode) -.-> log
        c1(audio.mp3.decode) -.-> log
        c2(audio.spatial) -.-> log
        c3(omni.core) -.-> log

        classDef bold font-weight:bold,stroke-width:4px;
        class log bold