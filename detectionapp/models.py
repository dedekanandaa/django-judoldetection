from django.db import models

class history(models.Model):
    id = models.AutoField(primary_key=True)
    date = models.DateTimeField(auto_now_add=True)
    type = models.CharField(max_length=15)

class result(models.Model):
    id = models.AutoField(primary_key=True)
    img = models.TextField(blank=True, null=True)
    text = models.TextField(blank=True, null=True)
    primary_factor = models.TextField(blank=True, null=True)
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
    def visual_confidence(self):
        """Calculate visual confidence based on dynamic prediction"""
        return abs(self.visual_feature - 0.5) * 200

    @property
    def semantic_confidence(self):
        """Calculate semantic confidence based on dynamic prediction"""
        return abs(self.semantic_feature - 0.5) * 200

    @property
    def combined_confidence(self):
        """Calculate combined confidence based on dynamic prediction"""
        return abs(self.combined_feature - 0.5) * 200

    @property
    def confidence_percent(self):
        """Calculate combined confidence based on dynamic prediction"""
        return abs(self.confidence - 0.5) * 200

