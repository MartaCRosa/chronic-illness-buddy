from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a secure key

    # Register blueprints or views
    from .views import main
    app.register_blueprint(main)

    return app
