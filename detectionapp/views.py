from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator

# Update progress_view to accept task_id
def progress_view(request, task_id):
    return render(request, 'app/progress.html', {'task_id': task_id})

# auth views
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            # Redirect to a success page.
            next_url = request.POST.get('next') or 'detect:home'  
            return redirect(next_url)
        else:
            # Return an 'invalid login' error message.
            messages.error(request, 'Username atau Password salah')
    if request.user.is_authenticated:
        return redirect('detect:home')
    return render(request, 'registration/login.html')


# page views
def home_view(request):
    return render(request, 'page/home.html')

def about_view(request):
    from .models import history
    # Ambil data chart
    hasil_data = history.chart_hasil_data()
    metode_data = history.chart_metode_data()

    return render(request, 'page/about.html', {"hasil_data": hasil_data, "metode_data": metode_data,})

@login_required
def logout_view(request):
    logout(request)
    return redirect('detect:login')

# app views
@login_required
def search_view(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        limit = int(request.POST.get('limit'))
        if not query:
            messages.error(request, 'Kata kunci tidak boleh kosong')
            return redirect('detect:search')
        if not limit:
            messages.error(request, 'Limit tidak boleh kosong')
            return redirect('detect:search')
        if len(query) < 3:
            messages.error(request, 'Kata kunci terlalu pendek')
            return redirect('detect:search')
        if len(query) > 100:
            messages.error(request, 'Kata kunci terlalu panjang')
            return redirect('detect:search')
        if limit > 30:
            limit = 30
        if limit < 1:
            limit = 1

        from detectionapp.coba import main_search
        task = main_search.delay(query=query, num_results=limit)
        
        # Redirect to progress page with task_id
        return redirect('detect:progress', task_id=task.id)
    else:     
        return render(request, 'app/search.html', {"input_type": "Kata Kunci"})

@login_required
def file_view(request):
    if request.method == 'POST':
        file = request.FILES.get('image')
        if not file:
            messages.error(request, 'File tidak boleh kosong')
            return redirect('detect:file')

        # Save the file first, then pass the path to Celery
        from detectionapp.coba import prepare_uploaded_image
        file_path = prepare_uploaded_image(file)
        
        # Call Celery task with file path instead of file object
        from detectionapp.coba import image_upload
        task = image_upload.delay(file_path)
        
        return redirect('detect:progress', task_id=task.id)
    else:
        return render(request, 'app/file.html', {"input_type": "Import Gambar"})

@login_required
def domain_view(request):
    if request.method == 'POST':
        domain = request.POST.getlist('domain')

        from detectionapp.coba import main_domain
        task = main_domain.delay(domain)
        return redirect('detect:progress', task_id=task.id)
    else: 
        limit = int(request.GET.get('limit', 5))
        if limit < 1:
            limit = 1
        elif limit > 10:
            limit = 10
        return render(request, 'app/domain.html', {"input_type": "Domain", "limit" : range(1, limit+1), "lim": limit})

# result views
@login_required
def history_view(request):
    from .models import history
    items = history.objects.all().order_by('-date')
    paginator = Paginator(items, 6)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)


    return render(
        request,
        'app/history.html',
        {
            "page_obj": page_obj,
        }
    )

@login_required
def result_view(request, id):
    from .models import result
    items = result.objects.filter(history_id=id)
    return render(request, 'app/result.html', {"data" : items})
