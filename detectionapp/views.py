from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages

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
    return render(request, 'page/about.html')

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
        if limit > 20:
            limit = 20
        if limit < 1:
            limit = 1

        from . import coba
        import asyncio
        result = asyncio.run(coba.analyze_for_django(query=query, num_results=limit))

        return redirect('detect:result', id=result)
    else:     
        return render(request, 'app/search.html', {"input_type": "Kata Kunci"})

@login_required
def file_view(request):
    return render(request, 'app/file.html', {"input_type": "Import Gambar"})

@login_required
def domain_view(request):
    if request.method == 'POST':
        return redirect('detect:history')
        # query = request.POST.get('query')
        # limit = int(request.POST.get('limit'))
        # if not query:
        #     messages.error(request, 'Kata kunci tidak boleh kosong')
        #     return redirect('detect:search')
        # if not limit:
        #     messages.error(request, 'Limit tidak boleh kosong')
        #     return redirect('detect:search')
        # if len(query) < 3:
        #     messages.error(request, 'Kata kunci terlalu pendek')
        #     return redirect('detect:search')
        # if len(query) > 100:
        #     messages.error(request, 'Kata kunci terlalu panjang')
        #     return redirect('detect:search')
        # if limit > 20:
        #     limit = 20
        # if limit < 1:
        #     limit = 1

        # from . import coba
        # import asyncio
        # result = asyncio.run(coba.analyze_for_django(query=query, num_results=limit))

        # return redirect('detect:result', id=result)
    else: 
        limit = int(request.GET.get('limit', 5))
        if limit < 1:
            limit = 1
        elif limit >= 20:
            limit = 10
        return render(request, 'app/domain.html', {"input_type": "Domain", "limit" : range(1, limit+1), "lim": limit})

# result views
@login_required
def history_view(request):
    from .models import history, result
    items = history.objects.all().order_by('-date').values()
    
    for item in items:
        item['result'] = result.objects.filter(history_id=item['id']).values()
        item['count'] = len(item['result'])
        item['judi'] = len([x for x in item['result'] if x['predict'] == 2])
        item['non_judi'] = len([x for x in item['result'] if x['predict'] == 3])
        if item['count'] > 0:
            item['result'] = item['result'][0]
        else:
            item['result'] = None

    return render(request, 'app/history.html', {"data" : items})

@login_required
def result_view(request, id):
    from .models import result
    items = result.objects.filter(history_id=id)
    return render(request, 'app/result.html', {"data" : items})
