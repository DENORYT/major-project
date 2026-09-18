import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';

class WebSocketService {
  WebSocketService._privateConstructor();
  static final WebSocketService instance = WebSocketService._privateConstructor();

  WebSocketChannel? _channel;
  Function(String, String?)? onWordDetected;
  Function(String)? onSentenceGenerated;

  void connect(String url) {
    _channel = WebSocketChannel.connect(Uri.parse(url));
    _channel!.stream.listen((message) {
      final data = jsonDecode(message);
      
      if (data['type'] == 'llm_result') {
        if (onSentenceGenerated != null) {
          onSentenceGenerated!(data['sentence']);
        }
      } else {
        String? prediction = data['prediction'];
        String? addedWord = data['added_word'];
        if (onWordDetected != null && prediction != null) {
          onWordDetected!(prediction, addedWord);
        }
      }
    }, onError: (error) {
      print('WebSocket error: $error');
    });
  }

  void sendFrame(String base64Image) {
    if (_channel != null) {
      _channel!.sink.add(jsonEncode({
        'action': 'process_frame',
        'image': base64Image,
      }));
    }
  }

  void requestSentence(List<String> words) {
    if (_channel != null) {
      _channel!.sink.add(jsonEncode({
        'action': 'generate_sentence',
        'words': words,
      }));
    }
  }

  void disconnect() {
    _channel?.sink.close();
  }
}
