using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Entity;
using System.Linq;
using System.Net;
using System.Web;
using System.Web.Mvc;
using KourseWork.Models;

namespace KoursWork.Controllers
{
    public class СведениеController : Controller
    {
        private AudioStudioDBEntities2 db = new AudioStudioDBEntities2();

        // GET: Сведение
        public ActionResult Index()
        {
            var сведение = db.Сведение.Include(с => с.Перечень_услуг);
            return View(сведение.ToList());
        }

        // GET: Сведение/Details/5
        public ActionResult Details(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Сведение сведение = db.Сведение.Find(id);
            if (сведение == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(сведение);
        }

        // GET: Сведение/Create
        public ActionResult Create()
        {
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование");
            return View();
        }

        // POST: Сведение/Create
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Create([Bind(Include = "Код_услуги,Кол_во_дорожек,Работа_в_днях,Плагин")] Сведение сведение)
        {
            if (ModelState.IsValid)
            {
                try{db.Сведение.Add(сведение);
                db.SaveChanges();
                return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка создания!";
                    return RedirectToAction("Error");
                }
            }

            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование", сведение.Код_услуги);
            return View(сведение);
        }

        // GET: Сведение/Edit/5
        public ActionResult Edit(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Сведение сведение = db.Сведение.Find(id);
            if (сведение == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование", сведение.Код_услуги);
            return View(сведение);
        }

        // POST: Сведение/Edit/5
        // Чтобы защититься от атак чрезмерной передачи данных, включите определенные свойства, для которых следует установить привязку. Дополнительные 
        // сведения см. в разделе https://go.microsoft.com/fwlink/?LinkId=317598.
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Edit([Bind(Include = "Код_услуги,Кол_во_дорожек,Работа_в_днях,Плагин")] Сведение сведение)
        {
            if (ModelState.IsValid)
            {
                try{db.Entry(сведение).State = EntityState.Modified;
                db.SaveChanges();
                return RedirectToAction("Index");
                }
                catch
                {
                    ViewBag.Data = "Ошибка изменения!";
                    return RedirectToAction("Error");
                }
            }
            ViewBag.Код_услуги = new SelectList(db.Перечень_услуг, "Код_услуги", "Оборудование", сведение.Код_услуги);
            return View(сведение);
        }

        // GET: Сведение/Delete/5
        public ActionResult Delete(string id)
        {
            if (id == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            Сведение сведение = db.Сведение.Find(id);
            if (сведение == null)
            {
                return RedirectToAction("Error404", "Error");
            }
            return View(сведение);
        }

        // POST: Сведение/Delete/5
        [HttpPost, ActionName("Delete")]
        [ValidateAntiForgeryToken]
        public ActionResult DeleteConfirmed(string id)
        {
            try{Сведение сведение = db.Сведение.Find(id);
            db.Сведение.Remove(сведение);
            db.SaveChanges();
            return RedirectToAction("Index");
            }
            catch
            {
                ViewBag.Data = "Ошибка удаления! Возможно, эти данные где-то есть...";
                return RedirectToAction("Error");
            }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                db.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
