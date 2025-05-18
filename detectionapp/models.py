from django.db import models

class history(models.Model):
    id = models.AutoField(primary_key=True)
    date = models.DateTimeField(auto_now_add=True)
    type = models.CharField(max_length=15)

class result(models.Model):
    class FactorChoice(models.IntegerChoices):
        semantic = 1, 'semantic'
        visual = 2, 'visual'
        combined = 3, 'combined'
    
    id = models.AutoField(primary_key=True)
    img = models.TextField(blank=True, null=True)
    text = models.TextField(blank=True, null=True)
    primary_factor = models.IntegerField(choices=FactorChoice, blank=True, null=True)
    visual_feature = models.FloatField(blank=True, null=True)
    semantic_feature = models.FloatField(blank=True, null=True)
    combined_feature = models.FloatField(blank=True, null=True)
    confidence = models.FloatField(blank=True, null=True)
    predict = models.BooleanField(blank=True, null=True)
    url = models.URLField(blank=True, null=True)
    history_id = models.ForeignKey(
        history,
        on_delete=models.CASCADE,
        related_name="results"
    )

    @property
    def primary_factor_label(self):
        return self.get_primary_factor_display()

    
