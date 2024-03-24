from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import Pdf


class PdfInline(admin.TabularInline):
  model = Pdf
  extra = 0  # No extra forms
  fields = ['pdf']


# Extend the existing UserAdmin to include the PdfInline
class UserAdmin(BaseUserAdmin):
  inlines = [PdfInline]

  # This method is optional, it's here to show you how to add a custom field in the list display
  def get_form(self, request, obj=None, **kwargs):
    form = super(UserAdmin, self).get_form(request, obj, **kwargs)
    is_superuser = request.user.is_superuser
    disabled_fields = set()  # type: Set[str]

    # Prevent non-superusers from editing superuser's profiles
    if not is_superuser:
      disabled_fields |= {
        'username',
        'is_superuser',
        'user_permissions',
      }

    # Prevent staff users from editing their own permissions
    if (
      not is_superuser
      and obj is not None
      and obj == request.user
    ):
      disabled_fields |= {
        'is_staff',
        'is_superuser',
        'groups',
        'user_permissions',
      }

    for f in disabled_fields:
      if f in form.base_fields:
        form.base_fields[f].disabled = True

    return form


# Unregister the original User admin and register the customized version
admin.site.unregister(User)
admin.site.register(User, UserAdmin)
