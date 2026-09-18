import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../services/websocket_service.dart';
import '../services/tts_service.dart';

class HomeScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const HomeScreen({super.key, required this.cameras});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  CameraController? _controller;
  bool _isStreaming = false;
  String _currentSign = "--";
  List<String> _accumulatedWords = [];
  bool _isConversationMode = false;

  // Change this to your laptop's IP address when running on a physical device.
  // Example: 'ws://192.168.1.100:8000/ws/stream'
  String _backendUrl = 'ws://10.0.2.2:8000/ws/stream';

  @override
  void initState() {
    super.initState();
    _initCamera();
    _initServices();
  }

  Future<void> _initServices() async {
    await TTSService.instance.init();
    
    WebSocketService.instance.onWordDetected = (prediction, addedWord) {
      setState(() {
        _currentSign = prediction;
      });
      
      if (addedWord != null) {
        if (!_isConversationMode) {
          // Word Mode: Speak instantly
          TTSService.instance.speak(addedWord);
        } else {
          // Conversation Mode: Accumulate
          setState(() {
            if (addedWord == 'SPACE') {
              // Usually handled in UI space, but let's keep it simple
            } else if (addedWord == 'DEL') {
              if (_accumulatedWords.isNotEmpty) {
                _accumulatedWords.removeLast();
              }
            } else {
              _accumulatedWords.add(addedWord);
            }
          });
        }
      }
    };

    WebSocketService.instance.onSentenceGenerated = (sentence) {
      TTSService.instance.speak(sentence);
      setState(() {
        _accumulatedWords.clear();
        _accumulatedWords.add(sentence); // Show the final generated sentence
      });
    };
  }

  Future<void> _initCamera() async {
    if (widget.cameras.isEmpty) return;
    
    // Use the front camera if available
    CameraDescription selectedCamera = widget.cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => widget.cameras.first,
    );

    _controller = CameraController(
      selectedCamera,
      ResolutionPreset.medium,
      enableAudio: false,
    );

    await _controller!.initialize();
    if (mounted) setState(() {});
  }

  void _toggleStreaming() {
    if (_isStreaming) {
      _controller?.stopImageStream();
      WebSocketService.instance.disconnect();
    } else {
      WebSocketService.instance.connect(_backendUrl);
      int frameCount = 0;
      _controller?.startImageStream((CameraImage image) {
        frameCount++;
        if (frameCount % 3 == 0) { // Throttle to ~10 fps for network stability
          // Note: In a production app, converting YUV420 to JPEG in Dart can be slow.
          // Native plugins or sending byte arrays directly is preferred. 
          // For now, this assumes a custom compression step or backend handling.
          // Since we are mocking the exact implementation detail here:
          // _sendFrameToBackend(image);
        }
      });
    }
    setState(() {
      _isStreaming = !_isStreaming;
    });
  }

  void _triggerAgenticLLM() {
    if (_accumulatedWords.isNotEmpty) {
      WebSocketService.instance.requestSentence(_accumulatedWords);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    WebSocketService.instance.disconnect();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ISL Communication Assistant'),
        actions: [
          Row(
            children: [
              const Text('Conv Mode'),
              Switch(
                value: _isConversationMode,
                onChanged: (val) {
                  setState(() {
                    _isConversationMode = val;
                    _accumulatedWords.clear();
                  });
                },
              ),
            ],
          )
        ],
      ),
      body: Column(
        children: [
          // Camera Preview
          Expanded(
            flex: 2,
            child: Container(
              color: Colors.black,
              child: _controller?.value.isInitialized == true
                  ? CameraPreview(_controller!)
                  : const Center(child: CircularProgressIndicator()),
            ),
          ),
          
          // Current Sign
          Container(
            padding: const EdgeInsets.all(8),
            color: Colors.teal.shade900,
            width: double.infinity,
            child: Text(
              'Detected: $_currentSign',
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
          ),

          // Output Box
          Expanded(
            flex: 1,
            child: Container(
              padding: const EdgeInsets.all(16),
              width: double.infinity,
              color: Colors.black87,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Translation:', style: TextStyle(color: Colors.grey)),
                  const SizedBox(height: 8),
                  Text(
                    _isConversationMode 
                        ? _accumulatedWords.join(' ')
                        : _currentSign,
                    style: const TextStyle(fontSize: 24, color: Colors.white),
                  ),
                ],
              ),
            ),
          ),

          // Controls
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  icon: Icon(_isStreaming ? Icons.stop : Icons.play_arrow),
                  label: Text(_isStreaming ? 'Stop AI' : 'Start AI'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isStreaming ? Colors.red : Colors.teal,
                  ),
                  onPressed: _toggleStreaming,
                ),
                if (_isConversationMode)
                  ElevatedButton.icon(
                    icon: const Icon(Icons.record_voice_over),
                    label: const Text('Speak Sentence'),
                    onPressed: _triggerAgenticLLM,
                  ),
              ],
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          showDialog(
            context: context,
            builder: (context) {
              TextEditingController ipController = TextEditingController(text: _backendUrl);
              return AlertDialog(
                title: const Text('Backend Server URL'),
                content: TextField(
                  controller: ipController,
                  decoration: const InputDecoration(hintText: "ws://192.168.X.X:8000/ws/stream"),
                ),
                actions: [
                  TextButton(
                    onPressed: () {
                      setState(() {
                        _backendUrl = ipController.text;
                      });
                      Navigator.pop(context);
                    },
                    child: const Text('Save'),
                  )
                ],
              );
            }
          );
        },
        child: const Icon(Icons.settings),
      ),
    );
  }
}
