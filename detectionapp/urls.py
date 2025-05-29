from django.urls import path

from detectionapp import views

app_name = "detect"

urlpatterns = [
    path("", view=views.home_view, name="home"),
    path("about/", view=views.about_view, name="about"),
    path("login/", view=views.login_view, name="login"),
    path("logout/", view=views.logout_view, name="logout"),
    path("search/", view=views.search_view, name="search"),
    path("result/<int:id>", view=views.result_view, name="result"),
    path("file/", view=views.file_view, name="file"),
    path("domain/", view=views.domain_view, name="domain"),
    path("history/", view=views.history_view, name="history"),
    path("progress/<str:task_id>", view=views.progress_view, name="progress"),
]
