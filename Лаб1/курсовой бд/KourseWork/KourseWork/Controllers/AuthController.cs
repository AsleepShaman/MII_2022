using KourseWork.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using System.Web.Security;

namespace KourseWork.Controllers
{
    public class AuthController : Controller
    {
        private AudioStudioDBEntities2 db = new AudioStudioDBEntities2();
        public ActionResult Login()
        {
            return View();
        }

        public ActionResult IndexUser()
        {
            return View();
        }
        public ActionResult IndexAdmin()
        {
            return View();
        }

        public ActionResult WithoutAuth()
        {
            return View();
        }
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Login(string Pasport, string Password)
        {
            var User = db.Autentification.FirstOrDefault(p => p.Логин == Pasport && p.Пароль == Password);
            if (User != null)
            {
                if (User.Роль == "admin")
                {
                    return RedirectToAction("Index", "Home");
                }
                else if (User.Роль == "user")
                {
                    return RedirectToAction("IndexUser", "Auth");
                }
                else
                {
                    ModelState.AddModelError("", "Пользователя с таким логином и паролем нет");

                }
            }
            return View();
        }
    }
}