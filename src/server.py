import os
import logging
import json
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from src.indexer import ProjectIndexer
from src.search import SearchManager
from src.model.orchestrator import get_orchestrator

logger = logging.getLogger(__name__)

def create_app(indexer: ProjectIndexer):
    app = Flask(__name__)
    CORS(app)
    
    indexer.initialize_storage()
    orchestrator = get_orchestrator()
    search_manager = SearchManager(indexer, orchestrator=orchestrator)

    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        logger.info("Pre-loading models at startup...")
        if orchestrator:
            orchestrator.load()
        if search_manager.embedding_model:
            search_manager.embedding_model.load()
        if search_manager.reranker:
            search_manager.reranker.load()
        logger.info("All models loaded and ready.")

    @app.route('/api/search', methods=['POST'])
    def search():
        data = request.json
        query = data.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        logger.info(f"Web Search Query: {query}")
        
        if search_manager.chroma.collection.count() == 0:
            return jsonify({
                "error": "Codebase not indexed",
                "needs_indexing": True
            }), 200

        try:
            search_data = search_manager.search(query, 10)
            results = search_data["results"]
            
            if not results:
                return jsonify({"answer": "No relevant code found.", "snippets": []})

            answer = search_manager.answer_query(query, results)
            
            formatted_snippets = []
            for res in results:
                s = res["snippet"]
                formatted_snippets.append({
                    "name": s.name,
                    "file_path": s.file_path,
                    "start_line": s.start_line + 1,
                    "content": s.content,
                    "summary": s.summary,
                    "relations": res.get("relations", [])
                })
                
            return jsonify({
                "answer": answer,
                "snippets": formatted_snippets,
                "hyde_used": search_data["hyde_used"],
                "final_query": search_data["final_query"]
            })
        except Exception as e:
            logger.error(f"Search error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/search/stream', methods=['POST'])
    def search_stream():
        data = request.json
        query = data.get('query')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        try:
            search_data = search_manager.search(query, 10)
            results = search_data["results"]
            
            if not results:
                return jsonify({"answer": "No relevant code found.", "snippets": []})

            formatted_snippets = []
            for res in results:
                s = res["snippet"]
                formatted_snippets.append({
                    "name": s.name,
                    "file_path": s.file_path,
                    "start_line": s.start_line + 1,
                    "content": s.content,
                    "summary": s.summary,
                    "relations": res.get("relations", [])
                })

            def generate():
                # First, send metadata and snippets
                initial_payload = {
                    "type": "metadata",
                    "snippets": formatted_snippets,
                    "hyde_used": search_data["hyde_used"],
                    "final_query": search_data["final_query"]
                }
                yield f"data: {json.dumps(initial_payload)}\n\n"

                for token in search_manager.stream_answer_query(query, results):
                    payload = {"type": "token", "token": token}
                    yield f"data: {json.dumps(payload)}\n\n"
                
                yield "data: [DONE]\n\n"

            return Response(stream_with_context(generate()), content_type='text/event-stream')
            
        except Exception as e:
            logger.error(f"Streaming search error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/reindex', methods=['POST'])
    def reindex():
        try:
            logger.info("Web Reindex Request started...")
            indexer.initialize_storage()
            snippets = indexer.extract_snippets()
            indexer.extract_relationships(snippets)
            indexer.summarize_snippets(snippets)
            embeddings = indexer.embed_snippets(snippets)
            indexer.save(snippets, [], embeddings=embeddings)
            indexer.cleanup(snippets)
            logger.info("Web Reindex Request completed successfully.")
            return jsonify({"status": "success", "message": f"Indexed {len(snippets)} snippets"})
        except Exception as e:
            logger.error(f"Reindex error: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/status', methods=['GET'])
    def status():
        try:
            count = search_manager.chroma.collection.count()
            return jsonify({
                "project_id": indexer.context.project_id,
                "indexed_snippets": count,
                "is_indexed": count > 0
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/')
    def index():
        return app.send_static_file('index.html')

    return app

def run_server(indexer: ProjectIndexer, port=5000):
    os.environ['FLASK_DEBUG'] = '0'
    
    app = create_app(indexer)
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    app.static_folder = static_dir
    
    logger.info(f"Starting Web Server on http://localhost:{port} with hot reload")
    
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

