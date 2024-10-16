from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.utils.timezone import now
from .models import Messages, Chat

@receiver(post_save, sender=Messages)
@receiver(post_delete, sender=Messages)
def update_chat_updated_at(sender, instance, **kwargs):
    chat = instance.chat
    chat.updated_at = now()
    chat.save()
